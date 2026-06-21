# Coco — Architecture Design (v2)

A conversational agent that learns from the conversations she has. Knowledge gets organized into **packets** (topic-scoped notes), retrieved on conversational relevance, and consolidated over time. Sits on top of a skills layer that Coco can also call.

This document is the source of truth for the conceptual design. Implementation choices (runtime, embedding model, libraries) are tracked separately.

**v2 change vs v1:** packets now have *multiple* topic facets (each with its own vector) and a separate *entity list* — making memory multi-entry-point rather than single-handle. Retrieval becomes a three-channel RRF (topic BM25, topic max-cosine, entity BM25). Write path gains a "new facet vs new packet" decision. v1 data is discarded (clean break).

---

## Core concepts

- **Packet** — a unit of knowledge organized around one or more related subjects. Mixed content: declarative facts, procedural know-how, conceptual material. The atomic unit of long-term memory.
- **Topic facet** — up to 10 words. One *way* a packet can be invoked. A packet has a list of these — each is a separate retrieval handle.
- **Entity** — a proper noun, name, or non-common noun that identifies subjects in a packet's content. Coco decides which are worth indexing at write time.
- **Session** — one conversation, with its own list of topics encountered and packets loaded.
- **Scratchpad** — short-term buffer of things mentioned once but not yet worth a packet. Stays single-topic for simplicity.
- **Strength** — a dynamic score per packet driven by use, governing both retrieval priority and how much detail surfaces.

The defining design choice: **retrieval surface ≠ content.** A packet is indexed by its topic facets and entities; content stays opaque to similarity search, leaving it free to be long, rich, and unstructured.

The v2 refinement: **multi-entry-point retrieval.** A packet about Alka is reachable from "Alka," "Alka's habits," "Alka in Delhi," or just the entity "Alka" appearing in conversation. One memory, many doors.

---

## Packet schema

```
Packet
  id
  topics:    [{text: str, vector: list}, ...]   ≤10 words each; multiple facets
  entities:  [str, ...]                          case-normalized strings (lowercased)
                                                  LLM-decided at write time
  content:
    gist              one line
    summary           one paragraph
    full              unstructured markdown + base64 images (images deferred)
  strength_events:    [(event_type, timestamp, weight), ...]
                      event_type ∈ {retrieval, use, write}
                      strength is per-packet, not per-facet
  created_at, updated_at, source_session_ids
```

Notes:
- A packet has *N* topic facets, each capturing a different "way to invoke" the packet. Facets accumulate over the packet's life via integrate-on-write.
- Entities are plain strings (no per-entity vectors). Matched via case-insensitive token presence — "Alka" matches "alka" but not "my wife." Embedding-based entity aliasing is deferred.
- `content` carries three fidelities side-by-side. Regenerated whenever content changes.
- `strength_events` is append-only. Strength is computed lazily with exponential decay.

## Session state

```
Session
  topics              [{topic_text, topic_vector, first_seen_turn, last_seen_turn}, ...]
                      grows over the session, no eviction within session
                      stays single-facet-per-turn for simplicity
  current_topic       pointer into topics
  loaded_packets      [{packet_id, slice_loaded}, ...]
                      persists for whole session once retrieved
```

- Session topics **reset** at session end. No cross-session topic continuity.
- A packet, once loaded, **stays loaded for the rest of the session**.

## Scratchpad schema (unchanged from v1)

```
ScratchpadEntry
  id
  topic                 ≤10 words (single topic)
  topic_vector          embedding
  raw_excerpts          conversation snippets where the topic was mentioned
  mention_count
  created_at, last_seen_at, sessions_seen
```

- Discarded after 10 sessions without re-mention (configurable).
- Same retrieval mechanic as packets (against the topic vector).
- On promotion to packet, the scratchpad's `topic` becomes the new packet's *first* topic facet. Additional facets accumulate over future integrate-on-write events.

---

## The turn loop

```
1. User speaks.

2. Pre-retrieval (added in v1 build, kept):
     - embed user message
     - run 3-channel RRF (see "Retrieval scoring") against packet store
     - load matching packets into session at strength-appropriate slice
     - increment retrieval_count on newly loaded packets

3. Coco's main LLM call (single prompt, structured output):
     - generates her reply (using loaded packets as her memory)
     - generates current ≤10-word topic facet for this turn
     - declares which loaded packets she actually drew from
       → increments use_count on each

4. Topic resolution against session topics:
     - cosine match current topic facet vs session.topics
     - if score ≥ topic_match_threshold → update existing topic, set current
     - else → add as new topic to session, set current
                  ↓
            triggers new-topic retrieval pass (step 5)

5. New-topic retrieval pass (refinement of step 2):
     - run 3-channel RRF using the LLM-refined topic facet as query
     - load any new matches above retrieval_threshold

6. Write-path decision (when this turn yields new knowledge):
     - if best-matching existing packet's max-cosine ≥ existing_packet_match_threshold:
         → add facet / integrate content into that packet (see "Integrate-on-write")
     - else if a near-duplicate scratchpad entry exists:
         → promote scratchpad entry → new packet (LLM extracts entities + initial facets)
     - else:
         → insert new scratchpad entry (single topic, raw excerpt)

7. Integrate-on-write LLM call (for the matched packet from step 6):
     - input: existing content + new content
     - output: detect conflict; if conflict → pause and ask user;
               otherwise emit: updated gist + summary + full,
                              updated topic-facets list (new facets dedup'd by vector),
                              updated entities list
     - increments write_count
```

---

## Retrieval scoring

Three RRF channels per packet:

```
Channel A — topic BM25         query text  vs  packet's combined topic-facets text
Channel B — topic max-cosine   query vector vs  max over packet.topics[*].vector
Channel C — entity BM25         query text  vs  packet's entities bag (lowercased)

RRF_score(packet) = Σ_channel 1 / (k + rank_in_channel)
                    k = 60 (default)

final_score(packet) = RRF_score + g(strength(packet))
```

- **Channel A** catches lexical overlap with topic phrases ("Alka's habits" hitting a packet with that facet text).
- **Channel B** is the multi-vector core — pairwise cosine across all topic facets, max wins. A packet with 5 facets gets *its best matching facet's* cosine into the rank.
- **Channel C** catches entity mentions — "Alka" appearing in the user's message hits any packet whose entity list contains "alka."

`g` is an additive strength bias scaled so strength influences ranking but cannot dominate a sharp semantic match.

Alternative methods (weighted sum, top-K then rerank) configurable but not default.

---

## Strength dynamics

Strength is computed lazily at read time from the event log, with exponential time decay:

```
strength(packet, now) = Σ over strength_events of:
                          weight(event_type) · 0.5^((now - event.timestamp) / half_life)

weights (config defaults):  retrieval = 1, use = 3, write = 5
half_life (config default): 30 days
```

Slice selection at retrieval time:
```
if strength < band_gist_max     → load gist
elif strength < band_summary_max → load summary
else                              → load full
```

Strength is **per packet, not per facet** — a packet is one memory, its facets are entry points, not separate memories.

---

## Skills layer

Coco can call skills (tools) just like any agent. Packets may reference skills in their content ("for upcoming birthdays, query the calendar skill"). When a packet referencing a skill is loaded, Coco has the option to invoke it. No special integration beyond convention.

---

## Observability — Langfuse tracing

Coco's behaviour depends on prompts, thresholds, and the embedding model — all of which need empirical tuning. Every LLM call and key memory operation is traced to **Langfuse** so failures (wrong packet loaded, missed conflict, over-aggressive promotion) can be inspected and replayed without sprinkling print statements through the code.

**Credentials via `.env`** (loaded by `python-dotenv` at process start):
```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted URL
```
If keys are absent, tracing is silently disabled (offline mode — Coco still runs).

**Trace shape — one trace per `chat_turn()` invocation.** Trace-level metadata: `session_id`, `session_n`, `turn_index`, raw user message.

### Langfuse sessions — grouping turns of a conversation

Every Coco session (one run of the CLI, one `Session()` instance) maps to a single Langfuse session. Every `chat_turn()` within it becomes a separate trace, but all those traces share the same Langfuse `session_id`. This makes the Langfuse UI show a whole conversation as one unit you can scroll through turn-by-turn, rather than as scattered, unrelated traces.

**Mechanism.** Implemented via `langfuse.propagate_attributes(session_id=..., user_id=...)` as an outer context manager wrapping each turn's root span. `propagate_attributes` uses OpenTelemetry baggage so the session/user assignment flows into all nested spans automatically — no per-span plumbing needed.

**Attribute mapping:**
| Langfuse attribute | Coco source | Purpose |
|---|---|---|
| `session_id` | `Session.id` (e.g. `ses_695a743fe08d`) | groups all turns of one Coco conversation |
| `user_id` | `config.user_name` | separates multiple users sharing one Langfuse project |
| `metadata.session_n` | the global session counter | lets the UI order sessions chronologically |
| `metadata.turn_index` | turn position within the session | orders traces inside the session view |

**What you see in Langfuse.** Open a session → see the ordered list of turn traces → click a turn → see its nested spans (pre_retrieval, main_reply, topic_resolution, refinement_retrieval, integrate_on_write, new_packet_creation) and the `retrieval`/`use`/`write` scores. Replay or A/B prompt changes per turn or across whole sessions.

**Spans inside each trace:**

| Span | Kind | Captures |
|---|---|---|
| `pre_retrieval` | span | user-message vector, candidates considered, per-channel RRF ranks, packets loaded with slice |
| `main_reply` | generation | full system prompt, framed user message, model, raw output, parsed JSON (reply / topic_facet / packets_used / new_knowledge) |
| `topic_resolution` | span | new topic-facet vector, session topics, matched index or "new" |
| `refinement_retrieval` | span | same as pre_retrieval, with the LLM-refined topic facet as query |
| `integrate_on_write` (per merge) | generation | existing facets/entities/content, new content, model, output (conflict flag, updated gist/summary/full, facets, entities) |
| `new_packet_creation` (on promote) | generation | seed topic, seed content, model, output (initial content + facets + entities) |

Strength events recorded on packets (retrieval / use / write) are surfaced as Langfuse `score`s on the relevant trace, so the post-hoc view shows which packets were retrieved, used, or written for each turn.

**Why this is in the doc, not deferred:** Coco's correctness criteria are observational (did the right packet load? did Coco use it?) — without tracing, evaluation is anecdotal. Tracing is part of the system, not optional instrumentation.

---

## Configuration knobs

```
topic_match_threshold              same-topic vs new-topic in session
retrieval_threshold                minimum final_score to load a packet
existing_packet_match_threshold    add facet to existing vs new packet  (default 0.60)
scratchpad_promote_threshold       duplicate detection in scratchpad
facet_dedup_threshold              cosine above which a new facet is treated as duplicate of existing
recency_window                     turns considered for current-topic       (default 5)
hybrid_search_method               RRF | weighted_sum | top_k_rerank        (default RRF)
hybrid_search_k                    RRF constant                              (default 60)
strength_weights                   {retrieval: 1, use: 3, write: 5}
strength_half_life_days            (default 30)
strength_additive_bias_scale       scale of g(strength) bias
band_gist_max, band_summary_max    slice band boundaries
scratchpad_discard_after_sessions  (default 10)
```

---

## Worked examples

**Alka (multi-entry-point).** A packet about Alka accumulates over conversations. After several sessions it might have:
- topics: `["Alka — wife of Shishir", "Alka's family background", "Alka's habits and preferences", "Alka in Delhi years"]`
- entities: `["alka", "shishir", "delhi", "priya"]`
- content: full prose covering all of the above

Mentioning just *"Alka"* in conversation hits Channel C (entity bag), retrieving the packet directly. Saying *"how did Alka and Shishir meet?"* hits Channel B via the "Alka — wife of Shishir" facet. Saying *"what does Alka think about..."* hits Channel B via the "Alka's habits and preferences" facet. One packet, many entry points.

**NCS practice deck (gating preserved).** A packet's facets: `["NCS strategy practice deck", "NCS practice X overview"]`; entities `["ncs", "practice X"]`. If the user mentions "practice X" without "NCS," the entity-channel hit on "practice X" alone may pull the packet in — but the topic-channel max-cosine is weaker because the conversation topic is just "practice X overview" (no NCS in the facet text). The packet still scores but lower than a packet about a *different* practice X. If user wants stricter gating, set `existing_packet_match_threshold` higher or rely on entity richness (more entities co-occurring → stronger signal).

**Family packet with multiple sub-themes.** One packet can hold:
- topics: `["Shishir's nuclear family", "Alka's family (in-laws)", "Arjun's family"]`
- entities span every person mentioned
- content interleaves the three sub-themes in a single narrative

A conversation about Alka's parents matches via the "Alka's family" facet — even though the same packet also covers Shishir's nuclear family and his brother's family.

---

## Design decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Retrieval surface | Topic facets + entities, not content | Content stays long/rich/unstructured; retrieval surface stays small and explicit |
| Packet handle shape (v1) | Single topic + single vector | Initial simplification; too restrictive in practice |
| Packet handle shape (v2) | **List of topic facets + list of entities** | Memory is associative, not categorical — one packet, many ways to invoke |
| Multi-vector matching | Max cosine across facets | Sharp; one matching facet is enough; rewards focused facets over diffuse overlap |
| Entity definition | LLM-decided at write time | More flexible than POS/NER; runs only at packet write, not per turn |
| Entity matching | Text-based, case-insensitive (lowercased) | Simple; embedding-based aliasing is v2 future work |
| Entity extraction cadence | Write-time only, not per turn | Avoids 2× LLM calls per turn; BM25 on entity bag catches user mentions at retrieval |
| Topic-facet per turn | One per turn from Coco | Multiple facets accumulate over conversations, not within one turn |
| Hybrid search | 3-channel RRF (topic BM25, topic max-cosine, entity BM25), k=60 | Robust to score-distribution differences; symmetric across channels |
| Content format | Unstructured markdown + base64 images | Fits factual, procedural, conceptual; no rigid schema |
| Content fidelity | Multi-fidelity (gist / summary / full) stored alongside | Pay regeneration cost on rare write, not frequent read |
| Content updates | Integrate-on-write via LLM merge | Cleaner than append-only; conflict detection pauses for user confirm |
| Conflict resolution | Coco asks user to confirm | Preserves user oversight on contested facts |
| New facet vs new packet | Cosine match ≥ `existing_packet_match_threshold` → add facet to existing; else scratchpad/new packet | Higher bar than session topic match — adding a facet is a commitment |
| Strength: per packet, not per facet | Facets are entry points, packet is the memory | Same slice-band logic applies to the whole packet |
| Scratchpad scope | Stays single-topic | Promotion fills first facet of the new packet; further facets accrue via integrate-on-write |
| Scratchpad promotion | 2 near-duplicates by topic vector → promote | Spaced-repetition-style consolidation |
| Scratchpad lifetime | 10 sessions without re-mention → discard | Bound growth |
| Strength signals | Weighted combination of retrieval, use, write counts | All three capture different aspects of relevance |
| Strength decay | Exponential, 30-day half-life default | Dormant packets gracefully demote |
| Strength's role in scoring | Additive bias (not multiplier) | Strength matters but sharp semantic match still surfaces weak packets |
| Slice bands | Fixed numeric thresholds in config | Predictable and debuggable |
| Retrieval cadence | Pre-retrieval each turn + refinement on new-topic detection | Pre-retrieval avoids one-turn lag; refinement uses LLM-cleaned topic |
| Loaded-packet eviction | None within session | Once loaded, packets stay available the whole session |
| Session continuity | session.topics resets at session end | Cross-session continuity comes from packets themselves |
| Topic generation | Bundled into Coco's main reply prompt as structured output | One LLM call per turn for reply + topic |
| v1 → v2 migration | Clean break, discard v1 data | v1 store had no production memory; clean rebuild is simpler |
| Observability | Langfuse tracing via `.env`-loaded credentials, every LLM call + retrieval as spans | Tuning depends on inspecting real conversations; print-based debugging won't scale; offline mode when keys absent |
| Trace grouping | One Coco session = one Langfuse session; each turn = one trace inside it (via `propagate_attributes`) | Conversations are the natural unit of inspection; scattered per-turn traces lose context |

---

## Open / deferred items

- **Image embedding.** Images live in `full` content as base64. Whether image content also contributes its own vector or entity-like handle is deferred.
- **Strength event compaction.** The `strength_events` log grows monotonically. Periodic compaction (collapse old events into a decayed scalar + reset event log) is future work.
- **Initial bootstrap.** Coco starts empty: no packets, no scratchpad. Basic facts (user's name, date) learned in the first conversation like anything else.
- **Skills layer details.** How exactly a packet references a skill, and whether Coco autonomously invokes vs. proposes-and-asks, is left to implementation.
- **Embedding-based entity aliasing.** Currently entities match by lowercased text only. "Alka" and "my wife" won't co-resolve. v2 future work.
- **Facet pruning.** Nothing currently caps the number of facets a packet can accumulate. If a packet drifts to 20+ facets and they get noisy, consider periodic facet consolidation via LLM call.
- **Runtime / embedding model / hybrid-search implementation.** All deferred to implementation planning.
