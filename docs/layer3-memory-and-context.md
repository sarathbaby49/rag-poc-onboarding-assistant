# Layer 3 — Context & Memory: the last two topics

> Presenter's deep-dive for two session topics:
> **(1) Short-term vs long-term memory in applications**, and
> **(2) When context length alone isn't enough.**
>
> Slides + doc live as Claude artifacts. Hands-on: **M1** (session window) and
> **M4** (semantic recall) in `EXERCISES.md` (`src/memory.py`, self-check
> `python -m checks.check_memory`). `JoineeProfile` is provided working; the
> summary-buffer pattern below is a concept, not a coded exercise.

---

## 0. The one-sentence version

An LLM is **stateless** — it remembers nothing between calls. Every bit of
"memory" your app appears to have is *you* choosing what text to put back into the
next prompt. Memory design is therefore **context-window budgeting**: what do I
re-send, in what form, so the model can act like it remembers — without going
broke, slow, or dumb?

```
        ┌─────────────────────────────────────────────┐
        │  The model is a pure function of its input   │
        │     answer = f(prompt)   — no hidden state    │
        └─────────────────────────────────────────────┘
   "Memory" = your code deciding what goes back into `prompt` next turn.
```

---

## 1. Short-term vs long-term memory

Two different jobs. They are not two words for the same thing, and a real
assistant runs **both at once**.

### Short-term (a.k.a. session / working memory)
- **What:** the turns of the *current* conversation.
- **Why:** so "how do I run it?" → "what about the tests?" works without the user
  repeating themselves.
- **Lives:** in process memory, for the life of one chat. Thrown away at the end.
- **Lookup:** by **recency** — the last N turns (see M1).
- **Failure mode:** grows without bound and eventually **overflows the context
  window** (topic 2).

### Long-term (a.k.a. persistent / profile memory)
- **What:** durable facts about the person and the world — role, team, preferences,
  progress, decisions.
- **Why:** so next week it can say "you finished setup; your first PR is next" —
  across sessions, machines, days.
- **Lives:** on disk, in a DB, or in a vector store. Reloaded on the next session.
- **Lookup:** by **key** (`JoineeProfile` — load the profile for `name`) or by
  **meaning** (M4 — retrieve the memories relevant to *this* question).
- **Failure mode:** goes **stale** or bloats; someone has to decide what's worth
  keeping and what to forget.

### Side by side

| | Short-term (session) | Long-term (profile / store) |
|---|---|---|
| Holds | the current chat's turns | facts, preferences, progress |
| Lives | in memory, this process | on disk / DB / vector store |
| Scope | one conversation | across days & sessions |
| Lookup | recency (last N turns) | by key (`JoineeProfile`) or by meaning (M4) |
| Grows via | every new turn | an explicit "save this" decision |
| Main risk | overflows the window | goes stale / needs curation |
| In this repo | `SessionMemory` (M1) | `JoineeProfile` (provided), `MemoryStore` (M4) |

### The bit people miss: **promotion**

The interesting engineering is the *bridge* between them — deciding what in the
live chat is worth keeping forever.

```
   Short-term (session)                    Long-term (durable)
   ┌───────────────────┐   promote ──►     ┌────────────────────┐
   │ user: I'm on the  │  "extract the     │ profile.team =     │
   │ payments team btw │   durable fact"   │   'payments'       │
   │ user: how do I …  │                   │ steps = [setup, …] │
   └───────────────────┘   ◄── recall      └────────────────────┘
                          "pull back the memory
                           relevant to this turn"
```

- **Promote** on signals: an explicit preference, a completed step, a stable fact.
  (In the repo, `JoineeProfile.record_step()` promotes progress and persists it.)
- **Recall** the other way: given the current question, fetch the handful of
  long-term memories that matter (M4) instead of dumping the whole profile in.
- **Forget** on purpose: TTLs, "last seen", or letting the user edit their profile.
  Long-term memory with no forgetting strategy rots.

**Rule of thumb:** *recency* is for the conversation, *relevance* is for
everything older. If the answer could be anywhere in a long history, don't scroll —
retrieve.

---

## 2. When context length alone isn't enough

The tempting non-answer: *"just use a bigger window / a long-context model and
stuff everything in."* It fails four ways.

### 2a. Why "just make it bigger" breaks

1. **Cost** — you pay per token, *per turn*. Re-sending a 50-turn history every
   turn is quadratic spend over a conversation. A 200k-token context sent 20 times
   is 4M tokens billed.
2. **Latency** — more input tokens = slower first token. The chat feels sluggish
   exactly as it gets long.
3. **"Lost in the middle"** — models attend best to the **start and end** of a
   long prompt; facts buried in the middle get missed even though they're
   *technically* in context. Bigger window, lower recall of the buried fact.
4. **There is always a limit** — finite window, infinite conversation. "Bigger"
   moves the wall; it doesn't remove it. And onboarding history, a codebase, and a
   week of Slack will always outgrow any window.

```
  Recall of a fact vs. where it sits in a long prompt
  high │██                                   ██
       │███                                 ███
       │████                               ████
       │ ████                             ████
   low │   ██████████ (the middle) ████████
       └───────────────────────────────────────►  position in context
        start            middle             end
```

The fix is not a bigger bucket. It's putting **less, better** text in front of the
model. Two durable techniques:

### 2b. Compress — rolling summary (concept)

Keep the last few turns verbatim; **fold everything older into a running summary**
that rides in front of them. Context stays bounded no matter how long the chat
runs, and nothing is *dropped* — it's *compressed*.

```
  Plain window (M1): forgets                Summary buffer: compresses
  ┌─────────────────────────┐              ┌─────────────────────────────┐
  │ turn 6                   │              │ SUMMARY: name=Sam; on        │
  │ turn 7                   │              │  payments; finished setup    │
  │ turn 8   ← "my name is   │              ├─────────────────────────────┤
  │           Sam" already   │              │ turn 6                       │
  │           scrolled off   │              │ turn 7                       │
  │           and is GONE    │              │ turn 8                       │
  └─────────────────────────┘              └─────────────────────────────┘
   bounded, but loses facts                  bounded AND keeps the gist
```

Trade-off: a summary is **lossy** — fine for gist ("we're debugging auth"), risky
for exact values (that error code, that config flag). So in practice you summarise
*and* keep a retrievable copy (2c). A typical implementation folds each evicted
turn into the summary with a cheap extractive rule, or an `llm.ask(...)` call for
fluent prose — same interface either way. (We keep this a concept in the session;
the coded exercise is the retrieval half, M4.)

### 2c. Retrieve — semantic memory (M4)

Don't send the history; **search** it. Embed every memory once, and at each turn
pull back only the few pieces closest in meaning to the current question. This is
**exactly the RAG retrieval from R1**, pointed at the conversation/profile instead
of the docs.

```
   query: "which team am I on?"
             │  embed
             ▼
   ┌───────────────────────────┐   cosine similarity   top-k
   │ "Sam is on payments"  ●───┼──────── 0.71 ─────────►  ✔ recalled
   │ "coffee is on floor 3" ●──┼──────── 0.12            (relevant, not recent)
   │ "standup is at 10am"  ●───┼──────── 0.09
   └───────────────────────────┘
```

The magic: the recalled memory is chosen by **relevance, not recency**. The
payments fact comes back even though it was said first and the standup line is
newer. That's the whole point of M4 — and why a sliding window can't do this job.

### 2d. The full toolkit (pick per constraint)

| Technique | Keeps | Good when | Cost | In repo |
|---|---|---|---|---|
| Full history | everything | short chats | grows unbounded | — |
| Sliding window | last N turns | recency is enough | flat, but forgets | **M1** (exercise) |
| Summary buffer | gist + last N | long single chats | flat, lossy | concept |
| Semantic recall | relevant K | huge / cross-session history | flat, needs a store | **M4** (exercise) |
| Profile / key store | curated facts | stable per-user truth | tiny | `JoineeProfile` (provided) |

Real systems **layer** these: a window for the last few turns + a summary of the
session + semantic recall over long-term memory + a small profile of hard facts.
Each keeps the prompt small; together they make the model feel like it remembers.

---

## 3. How this maps to the code

Two hands-on exercises, kept deliberately light so there's time for agents (Layer 4):

| | Concept it makes concrete | Where | Status |
|---|---|---|---|
| **M1** | short-term memory, recency window, the overflow problem | `SessionMemory.as_messages` | exercise |
| **M4** | short-term vs long-term → **retrieve** by meaning, not recency | `MemoryStore.recall` | exercise |
| — | long-term memory, persistence across restarts | `JoineeProfile.save/.load` | provided |
| — | *context isn't enough* → **compress** (rolling summary) | — | concept only |

Self-check the two exercises: `python -m checks.check_memory`. Reference answers in
`solutions/memory.py` (try first!).

---

## 4. Talking points / likely questions

- **"Isn't a summary just lossy compression?"** Yes — that's the trade. Summarise
  for gist, keep a retrievable copy (M4) for exact values. Don't summarise numbers
  you'll need verbatim.
- **"Won't long-context models make this obsolete?"** They raise the ceiling, not
  remove it. Cost, latency, and lost-in-the-middle remain; conversations and
  corpora are unbounded. Retrieval + summary stay cheaper and often *more* accurate.
- **"Where does memory actually live?"** Session: RAM, this process. Long-term:
  a file (`JoineeProfile`), a DB, or a vector store (M4). The model itself stores nothing.
- **"How is M4 different from the doc retrieval we already built?"** It isn't,
  mechanically — same embed + cosine. The only difference is *what* you index: the
  conversation and the person, instead of the company docs. Memory is retrieval.
- **"How do you decide what to promote to long-term?"** Explicit signals
  (preferences, completed steps, stable facts), or an LLM "is this worth
  remembering?" pass. And pair it with a forgetting strategy or it rots.

---

## 5. Ninety-second recap

Short-term = the live chat, by recency. Long-term = who they are, by key or by
meaning. A bigger window is not the fix — it's costly, slow, and buries the
answer. **Compress** what scrolled off (summary — a concept here) and **retrieve**
what's relevant (semantic recall, **M4**). Layer them, and a stateless model feels
like it remembers you.
