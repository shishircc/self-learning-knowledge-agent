# Coco — Architecture Design (v2)

A conversational agent that learns from the conversations she has. Knowledge gets organized into **packets** (topic-scoped notes), retrieved on conversational relevance, and consolidated over time. Sits on top of a skills layer that Coco can also call.

This document is the source of truth for the conceptual design. Implementation choices (runtime, embedding model, libraries) are tracked separately.

**v2 change vs v1:** packets now have *multiple* topic facets (each with its own vector) and a separate *entity list* — making memory multi-entry-point rather than single-handle. Retrieval becomes a three-channel RRF (topic BM25, topic max-cosine, entity BM25). Write path gains a "new facet vs new packet" decision. Coco can also **ingest URLs** as conversational content — the user says "read this" with a link and the fetched page flows through the normal write path, with content-bearing images embedded inline as base64. Coco also **uploads documents** (PDF, DOCX, PPTX) as a third source of knowledge — the file is read in streaming fashion, the LLM judges whether it's a word-processing or presentation-style document, and content is split paragraph-by-paragraph (or slide-by-slide) into chunks that each route through the normal write path. Coco also gains **identity-aware multi-user operation**: anonymous or SSO (Microsoft Entra / Google) login at startup; roles carry a *capability set* (binary access checks) and a scalar *authoritativeness* (trust); every packet records source provenance (URL, speaker, or uploaded file) so conflict resolution and retrieval ranking can prefer higher-trust knowledge. Coco now also runs a **packet-anchored reply policy**: every substantive answer must be *anchored* by at least one fact in a loaded packet. Base-model / general knowledge may only be used as **connective tissue** to reason from a packet fact toward a conclusion — never as the standalone source of an answer. When *no* loaded packet is even relevant to the question, Coco's reply is exactly: **"I do not know about this."** (optionally followed by one short line offering to learn). Coco is a memory-anchored assistant: the packet supplies the substrate; reasoning may extend it. v1 data is discarded (clean break).

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
    full              unstructured markdown; image references use a custom URI
                       scheme: ![alt](coco-img:<image_id>)
  images:    [PacketImage, ...]                 base64-encoded images bound to this packet
  sources:   [PacketSource, ...]                 provenance of every write into this packet
                                                  (URL or conversation; append-only)
  authoritativeness: float                       max effective trust across sources ∈ [0,1]
  strength_events:    [(event_type, timestamp, weight), ...]
                      event_type ∈ {retrieval, use, write}
                      strength is per-packet, not per-facet
  created_at, updated_at, source_session_ids

PacketSource
  type:                "url" | "conversation" | "document"
  url:                 str | None                URL after redirects (type=url)
  domain_authoritativeness: float | None         resolved from config; None for conversation/document
  filename:            str | None                original filename            (type=document)
  document_type:       str | None                "word_processing" | "presentation"  (type=document)
  page_number:         int | None                1-based page index          (type=document)
  paragraph_index:     int | None                0-based paragraph within page (type=document, WP)
  file_authoritativeness:  float | None          resolved from config; None for url/conversation
  speaker_name:        str | None                from Identity.name   (all types: who introduced this)
  speaker_email:       str | None                from Identity.email  (all types)
  speaker_role:        str | None                from Identity.role   (all types)
  role_authoritativeness:      float             writer's role authoritativeness at write time
  effective_authoritativeness: float             max(role_auth, domain_auth or file_auth or 0)
  recorded_at:         ISO-8601

PacketImage
  id           "img_<hex>"                       globally unique
  alt          str | None                        descriptive text from the source page
  mime         "image/png" | "image/jpeg" | ...
  data_b64     str                                base64-encoded bytes (no data: prefix)
  dimensions   [w, h]                             post-downscale
  source_url   str | None                         the URL the image was fetched from
  added_at     ISO-8601
```

Notes:
- A packet has *N* topic facets, each capturing a different "way to invoke" the packet. Facets accumulate over the packet's life via integrate-on-write.
- Entities are plain strings (no per-entity vectors). Matched via case-insensitive token presence — "Alka" matches "alka" but not "my wife." Embedding-based entity aliasing is deferred.
- `content` carries three fidelities side-by-side. Regenerated whenever content changes.
- `content.full` references images by a custom URI: `![alt](coco-img:img_<id>)`. The base64 bytes live in `packet.images`, not inline. This keeps the markdown small enough to read, diff, and ship through prompts as text while images stay first-class objects.
- `images` is the authoritative list. Adding or removing an image always goes through this list; the markdown references resolve against it at load time. Orphan references (no matching image) and orphan images (no reference in `content.full`) are tolerated but logged.
- `sources` records the provenance of every write into the packet. Each entry is a `PacketSource` (URL or conversation) carrying the source identity and the trust scalars resolved at write time. Used to (a) route repeat ingests of the same URL to the same packet, (b) drive conflict resolution during integrate-on-write (higher-trust source wins), (c) bias retrieval ranking by `packet.authoritativeness`, and (d) surface provenance in replies ("I read this on Wikipedia last week — here's what changed").
- `authoritativeness` is the packet-level trust scalar, derived as `max(source.effective_authoritativeness for source in sources)`. Updated whenever a new source is appended. Never decreases — high-trust corroboration sticks even if a later low-trust write touches the packet.
- `strength_events` is append-only. Strength is computed lazily with exponential decay.

## Images as first-class packet content

A packet's images live in `packet.images` as a list of `PacketImage` records (id + base64 + alt + dimensions + source URL). `content.full` references them by id via markdown image syntax: `![alt](coco-img:img_<id>)`. This separation matters for three reasons:

1. **Text stays diff-able.** The markdown in `content.full` is short enough to inspect, edit, and pass through text-only LLM stages (integrate-on-write, new-packet creation). Base64 blobs would make a single packet's `full` content tens of thousands of tokens of unreadable noise.
2. **Images are loadable.** Because the bytes are stored as proper records on the packet, the agent can attach them as multimodal content blocks when the packet is loaded into the LLM's context — the LLM actually *sees* the chart or photo, not just an alt-text approximation.
3. **Independent lifecycle.** Images can be added, removed, or replaced without touching the surrounding text. The integrate-on-write LLM only manipulates references (image ids appear in the alt-text manifest it sees); the bytes carry through unchanged.

**Loading into LLM context.** When a packet is loaded with the `full` slice, its `PacketImage`s are attached to the next LLM call as `image` content blocks alongside the text content. The system prompt's packet rendering lists each image as `[img_<id>] alt="…" 124KB png 800x600` so the LLM can correlate alt text with image blocks. For `gist` and `summary` slices, images are not attached — only the text is loaded. This keeps the cheap slices cheap.

**Reference-vs-image consistency.** Markdown references in `content.full` and the actual `packet.images` list can drift in two ways:
- *Orphan reference* — `coco-img:img_X` appears in `content.full` but no image with that id exists. Renderers and the loader treat it as a missing image (logged, content otherwise preserved).
- *Orphan image* — an entry in `packet.images` is not referenced anywhere in `content.full`. Tolerated; happens when integrate-on-write decides an image is no longer worth surfacing in the prose. Periodic garbage-collection of orphan images is future work.

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
     - generates her reply, subject to the **packet-anchored reply policy**
       (see "Grounded reply policy" below): every substantive answer must
       be anchored by a fact from a loaded packet. General knowledge may
       be used to reason FROM a packet fact toward a conclusion, but not
       as the standalone source. If no loaded packet is relevant to the
       question, the reply is exactly "I do not know about this."
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

## Grounded reply policy

Coco is a **memory-anchored** assistant. Every substantive answer she gives must be *anchored* by at least one fact in a loaded packet. General knowledge may be used as **connective tissue** to reason from a packet fact toward a conclusion — it may not be used as the standalone source of an answer. If no loaded packet is relevant to the question, Coco refuses with an exact phrase.

**The rule, phrased tightly:**

> A substantive reply must be *anchored* by at least one loaded-packet fact. Base-model / general knowledge is permitted only when it is used to *reason FROM* an anchored packet fact toward the answer. When no loaded packet is relevant to the question at all, Coco's reply is exactly: **"I do not know about this."**

She may follow that refusal with one short, optional line offering a productive next step ("You can tell me and I'll remember, or share a URL/file and I'll read it."). She does not attempt to answer.

### What "packet-anchored" means

The load-bearing test: **can Coco name a specific packet-fact that contributes significantly to her answer?** "Significantly" means the packet fact does real work in the reasoning — take it away and the answer collapses. If yes, the answer is anchored. If no — if the answer would be the same even without the packet — it's ungrounded, and she refuses.

Coco may:

- **Quote or paraphrase** any content from loaded packets.
- **Synthesize across loaded packets** — if packet A says X and packet B says Y, combine them.
- **Reason within loaded content** — affirm implications a packet directly states.
- **Reason FROM a packet fact using general knowledge as connective tissue** — combine a packet fact with common-knowledge premises to derive a further conclusion, including conclusions that *negate* what the user asked. Example: if a packet says "diamond is hard enough to cut glass and steel" and the user asks "can glass cut diamond?", she may reason from the packet fact (diamond cuts glass) via the general principle (harder materials cut softer ones — so if diamond cuts glass, diamond is harder than glass) to conclude "no, glass cannot cut diamond." The packet fact is the anchor and contributes significantly to the answer; general knowledge bridges it to the conclusion. When the packet-fact and general knowledge can be combined so that **the packet contributes significantly** to the answer, Coco may answer.

Coco must NOT:

- **Answer from base knowledge alone.** If there is no relevant loaded packet, no reasoning bridge saves the answer. Refuse.
- **Smuggle an ungrounded claim through a reasoning step.** The reasoning must genuinely start from a loaded packet fact — not invoke a packet as a decorative preamble to a base-model answer.
- **Guess or hedge.** "I'm not sure but maybe…" is not permitted. Either the packet anchors an answer (with reasoning if needed) or Coco refuses.
- **Answer questions that share only a keyword with a packet.** A packet mentioning "Alka" doesn't authorize answers about arbitrary questions that happen to name Alka; the packet must actually contribute a fact that the reasoning uses.

**Transparency when the reasoning is non-trivial.** When the answer comes from reasoning FROM a packet fact rather than directly stated, Coco should make the chain visible: "The packet says X. Given Y, it follows that Z." This makes the anchor auditable and lets the user correct her if the general-knowledge bridge is wrong.

### Worked examples

*(All assume the named packets are loaded in the session.)*

- **Direct quote.**
  Packet: `"Alka lives in Delhi."` Q: *"Where does Alka live?"* → A: "Alka lives in Delhi." (Anchored: quoted.)

- **Synthesis across two packets.**
  Packet A: `"Alka lives in Delhi."` Packet B: `"Shishir works in Bangalore."` Q: *"Where do Alka and Shishir live/work?"* → A: "Alka lives in Delhi and Shishir works in Bangalore." (Anchored: both packets.)

- **Reasoning FROM a packet fact via a general principle (negation).**
  Packet: `"Diamond is hard enough to cut glass and steel."` Q: *"Can glass cut diamond?"* → A: "No — the packet says diamond cuts glass, which means diamond must be harder than glass. Harder materials cut softer ones, not the other way around, so glass cannot cut diamond." (Anchored: the diamond-cuts-glass fact. Bridge: the general principle that harder cuts softer, applied in reverse. The packet contributes *significantly* — without it there would be no established comparison between glass's hardness and diamond's, and no answer either way.)

- **Partial-coverage split.**
  Packet: `"Alka lives in Delhi."` Q: *"Does Alka like coffee?"* → A: "I know Alka lives in Delhi. I do not know about her coffee preferences." (Anchored on the covered part; refused on the uncovered part.)

- **Ungrounded — no relevant packet.**
  Loaded packets: (none about France or European geography.) Q: *"What's the capital of France?"* → A: "I do not know about this."

- **Ungrounded — keyword collision doesn't count as anchor.**
  Packet: `"Alka has a diamond ring."` Q: *"What's the chemical formula of diamond?"* → A: "I do not know about this." (The packet mentions the word "diamond" but contributes no fact that the answer would use — the ring is not the anchor for the chemical-formula question.)

- **Reasoning that just decorates a base-model answer — not allowed.**
  Packet: `"Alka lives in Delhi."` Q: *"What's the population of Delhi?"* → A: "I do not know about this." (Naming Delhi in the answer doesn't make the population fact grounded — the packet does not contribute anything the reasoning uses. The population claim would come entirely from base knowledge.)

### The exceptions carve-out

A small, closed list of things Coco may say without a packet backing them:

1. **The user's identity.** Coco knows the user's name (and email) from SSO login — this is present in her system prompt every turn (see "Identity in the agent's context"). She may address the user by name, refer to them naturally, and answer meta-questions like "what's my name?" from the identity block.
2. **Coco's own self-description.** Coco may explain who she is, what she does, and how she works ("I'm a memory-only assistant. I only answer from packets I've built up over our conversations."). This is who-she-is talk, not a knowledge claim about the world.
3. **Conversational niceties.** "Hi", "thanks", "you're welcome", "goodbye" and similar — these are not knowledge claims and are exempt.
4. **Introspection over currently-loaded state.** Coco may accurately describe *what she knows right now*: "I have packets about Alka, Shishir, and Delhi loaded — what would you like to talk about?" This is a statement about her own state, not about the world.
5. **Ingest and upload interactions.** When the user shares a URL or file with an ingest verb, Coco reads the content and can summarize it in that same turn — the fetched/uploaded content is her source for that reply (via `new_knowledge` items being routed into memory). The reply is grounded in the *just-read source*, which is functionally a fresh packet-in-flight.
6. **Clarification questions.** Coco may ask the user for clarification without herself making a knowledge claim ("What do you mean by X?" is a question, not an answer).

Anything not in this list is subject to the strict grounded-reply rule.

### Interaction with retrieval

Retrieval runs *before* the reply LLM (pre-retrieval on partials, refinement retrieval on submit). By the time Coco composes her reply, the session has already loaded whatever packets the 3-channel RRF surfaced above the retrieval threshold. So the answer to "did retrieval load anything?" is knowable at reply time.

- If retrieval loaded a packet that anchors the answer (directly or via reasoning) → Coco answers, making the reasoning chain visible if it's non-trivial.
- If retrieval loaded packets that are related but don't anchor the specific question → Coco names what IS covered and refuses the uncovered part.
- If retrieval loaded no packets that can anchor the question → Coco refuses with the exact phrase.

The refusal is **not a signal that retrieval failed** — sometimes there is genuinely no packet, and the honest answer is that Coco doesn't know. Retrieval tuning cannot make Coco know things she has never been taught.

### Refusal shape

The exact refusal template:

```
I do not know about this.
```

Optionally followed by ONE short line, when relevant:

```
You can tell me and I'll remember, or share a URL / file for me to read.
```

Not permitted in the refusal:

- Attempting a partial answer ("but I think it might be…").
- Speculation dressed as caveat ("I'm not sure, but…").
- Restating the user's question back with an apologetic wrapper.
- Making up related information from base knowledge to seem helpful.

Coco is polite, terse, and honest. Refusal is normal. Learning is her main job.

### The write path still runs on refusal turns

A refusal is not a dead turn. If the user's message contained substantive information (a new fact, a name, a claim), it still flows through the write path — the scratchpad or a new packet still gets it. Refusing to *answer* does not mean refusing to *learn*. This is how Coco grows: every time she can't answer, she has an opportunity to add a packet so that next time she can.

### Why this rule

Coco's value is *her own accumulated memory*, not general LLM competence. If she is allowed to fall back to base-model knowledge whenever a packet is missing, three things go wrong:

1. **The user can't tell what Coco actually remembers.** Every answer looks equally confident, whether it came from a packet Coco built up over months or from her base pre-training. The system loses its epistemic honesty.
2. **The write-path pressure disappears.** If Coco can already answer, why teach her? The whole "self-learning" loop only holds together when unanswered questions become the input to new packets.
3. **Provenance is a lie.** Packets carry `PacketSource` records precisely so an answer can be traced back to its origin. An unmarked base-model answer has no provenance and defeats the trust system (`role_authoritativeness`, `domain_authoritativeness`, `file_authoritativeness`).

Strict grounding makes Coco *legible* — a user always knows whether a claim came from her memory or from nowhere. There is never a fourth option.

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

## URL ingestion

Coco can read websites into her memory the same way she absorbs anything from conversation. The user shares a link with an explicit verb ("read this", "remember this site", "add this to your knowledge") and Coco fetches the page, converts it to markdown, decides which images are content-bearing, and lets the page flow through the normal write path.

The framing is deliberately conversational: an ingested URL is not a special "document" in a separate store — it produces ordinary `new_knowledge` candidates that route through the same integrate-vs-new-packet decision (step 6 of the turn loop). One source URL can land entirely in one existing packet (because its topic already covers the page), or split across packets, or create a fresh packet. Source URLs are tracked per packet so Coco knows what she has already read.

**Trigger.** The streaming small LM (`extraction`) gets two extra output fields: `is_ingest_request: bool` and `urls: list[str]`. The user has to express intent — Coco does not auto-fetch arbitrary links that happen to appear in conversation. Patterns the small LM is trained to recognize: "read this", "go read", "ingest", "remember this site", "learn from", "add this to your knowledge."

**Pipeline (in `chat_turn` when `is_ingest_request` is true):**

```
1. For each URL:
   - HTTP GET (no JS rendering; static fetch only in v1)
   - readability-lxml extracts the main article HTML
   - Walk <img> tags → collect (src, alt, intrinsic_dims) into an image manifest
   - markdownify converts the article HTML to markdown; <img> become [IMG_n] placeholders
   - Fetch each candidate image; downscale if longest edge > image_max_dim;
     drop if post-downscale size > image_max_bytes
2. Frame the main reply prompt with the fetched markdown + the candidate images
   themselves as multimodal content blocks:
   - The system prompt explains ingest mode and the [IMG_n] placeholder protocol.
   - The user message carries the fetched markdown text PLUS each candidate
     image as an image content block, in order, so the reply LLM (Sonnet) can
     visually inspect each one and decide whether it's content-bearing.
   - The LLM is asked to (a) reply conversationally with a brief ack + 2-3
     takeaways, and (b) emit `new_knowledge` items whose markdown content
     references the [IMG_n] tokens it wants to keep.
3. After the reply, for each kept [IMG_n]:
   - Mint a new globally-unique `img_<hex>` id.
   - Build a `PacketImage` record from the candidate's bytes + alt + dims + source_url.
   - Rewrite the [IMG_n] token in `new_knowledge[i].content` to
     `![alt](coco-img:img_<new_id>)`.
4. Run the standard write-path (step 6 of the turn loop) on each new_knowledge item:
   - If a packet's `sources` already contains a URL source for this URL → that
     packet is the prime integration target (overrides facet-cosine winner).
   - Otherwise, best-facet match decides integrate-vs-new exactly as for any
     other new_knowledge. (Ingest skips the scratchpad-only outcome — see §5.7.)
   - On every commit to a packet:
     - The `PacketImage` records for kept images are appended to packet.images.
     - A `PacketSource` (type=url) is appended to packet.sources, carrying the URL,
       the resolved `domain_authoritativeness`, the writer's role + role auth,
       and the computed `effective_authoritativeness = max(role_auth, domain_auth)`.
     - packet.authoritativeness is bumped to max(existing, new_source.effective_auth).
```

**Why the LLM decides which images to keep, with bytes in hand.** Pages are full of decorative images (icons, spacers, avatars, social-share buttons). Alt-text alone is unreliable — many decorative images carry expressive alt text, and many content-bearing diagrams have empty or generic alt. Passing the actual image bytes to the reply LLM as multimodal blocks lets it judge by appearance: a chart with axis labels survives; a stock hero photo doesn't. Cost is one extra image-input block per candidate (capped by `ingest_max_images_per_page`).

**Why images go to `packet.images`, not inline base64.** Inline base64 in `content.full` bloats the markdown by 4×—10× the raw byte size, makes integrate-on-write LLM prompts unreadable and ruinously expensive, and forces every text manipulation (gist generation, summary regen) to either pass huge strings around or risk truncation. The split is a cleaner separation of concerns: text talks about images via references, the image bytes live in a structured list, and the loader knows how to attach the bytes as multimodal content blocks when the packet is loaded with the `full` slice.

**Why no JS rendering in v1.** Most knowledge-bearing pages Coco will read (blogs, docs, Wikipedia, news, GitHub READMEs, papers) serve usable HTML statically. JS-only pages (Notion, some Substack, SPAs) will degrade with a "I couldn't find readable content" reply — deferred to a future Playwright fallback.

**Why the same write path.** Coco's value is in *associative* memory — multi-entry-point packets that grow over time. If ingestion bypassed the write path with its own "one packet per URL" rule, the user would end up with a packet for "Alka — Wikipedia" separate from the existing "Alka" packet, instead of the Wikipedia content enriching the packet she already has. Reusing integrate-on-write keeps memory consolidated.

---

## Document ingestion

Coco accepts file uploads (PDF, DOCX, PPTX, plain text / markdown) as a third source of knowledge, alongside conversation and URLs. The flow mirrors URL ingestion philosophically — same conversational verb ("read this file", "upload this", "add this document"), same `new_knowledge` write-path — but differs in three ways the conceptual model has to make room for:

1. **Streaming.** A 100-page PDF cannot be shoved into one LLM call. The file is read **page-by-page** as a stream; packets are written as soon as the next batch of content is ready, not after the whole document is parsed.
2. **Document-type detection.** PDFs come in two flavours that need different chunking: word-processing-style (long prose paragraphs, narrative flow) and presentation-style (slide layout, short bullets, one self-contained idea per page). The first few pages of extracted text are sent to the small LM, which classifies the document as `word_processing` or `presentation`. Native DOCX → always `word_processing`; native PPTX → always `presentation`; the classifier only fires when the format is ambiguous (PDF). Once classified, the chunking strategy is locked for the rest of the document.
3. **Paragraph-level routing.** Where URL ingestion processes one whole page as one LLM call, document ingestion splits content into *chunks* (paragraphs for WP, slides for presentations) and routes **each chunk individually** through the write-path. One PDF can produce many packets, or extend many existing packets, or both — knowledge from page 3 about "Alka" merges into the existing "Alka" packet while page 17 on "Delhi" creates a new packet. This finer granularity is what makes documents associative with the rest of memory rather than dropping in as one monolithic packet per file.

**The framing is conversational.** An uploaded document is not a special "document" in a separate store. It produces ordinary `new_knowledge` candidates that route through the same integrate-vs-new-packet decision Coco uses for every other write. Per-write provenance lands in `packet.sources` as `PacketSource(type="document", filename, page_number, paragraph_index, document_type, ...)` so a packet about Alka enriched from three different documents carries an honest audit trail of which paragraph in which file contributed what.

**Pipeline (when the user uploads a file with explicit intent — "read this", "upload this", "add this to your knowledge"):**

```
1. Identify the file. Format inferred from extension; MIME sniffed as a backstop.
   Reject unsupported formats with a brief reply ("I can read PDF / DOCX / PPTX /
   text / markdown right now"). Resolve file_authoritativeness from config
   (filename glob + path prefix; longest-match wins; default = 0.5).

2. Open the file. Use the format-appropriate streaming reader:
     - PDF      → pypdf, yields one page of text at a time.
     - DOCX     → python-docx, yields paragraphs in order.
     - PPTX     → python-pptx, yields slides (one chunk per slide).
     - .txt/.md → buffered file read, split into paragraph chunks.

3. Detect document type (PDF only; other formats imply the type).
   Read the first ~3 pages. Small LM call: classify as
   "word_processing" or "presentation". Lock for the rest of the document.

4. Chunk:
     - word_processing → split each page's text into paragraphs
       (double-newline boundaries; merge tiny paragraphs with neighbours;
        split paragraphs > max_paragraph_chars at sentence boundaries).
     - presentation   → one chunk per page/slide.
   Yield chunks lazily — the reader does not have to finish the file before
   chunking starts.

5. Stream chunks into the main reply LLM in batches (e.g. 10 chunks).
   Each batch produces:
     - A short progress reply ("processed pages 1-3 — saved 2 packets, updated 1")
       streamed to stdout so the user sees forward motion.
     - A list of new_knowledge items, each tagged with the originating chunk's
       paragraph_index + page_number + document_type.

6. Route each new_knowledge item through the standard write-path
   (best-facet match → integrate vs new packet; ingest skips the
   scratchpad-only outcome, same as URL ingest).
   Each commit appends a PacketSource(type="document", ...) carrying the
   chunk's coordinates.

7. After the stream completes, surface a final summary:
   "Read N pages of <filename>. Created K packets, updated M, skipped P filler chunks."
```

**Document-type detection rationale.** Word-processing prose and presentation slides chunk *very* differently. A 60-word slide is a complete idea on its own (one packet candidate); a 60-word paragraph in a book chapter is part of a larger argument. Forcing one chunking rule on both gives bad packets either way — slides get conflated, prose paragraphs get fragmented. Letting the small LM make this call early is much cheaper than running both strategies and reconciling.

**Why paragraph-level granularity, not whole-document.** A single LLM call over a whole PDF would be cheaper but loses the per-paragraph routing that lets one document enrich many existing packets. The compromise is to batch (10ish chunks per LLM call) instead of doing one call per paragraph — the LLM still decides per-paragraph (which integrates, which is filler, which creates a new packet), but the per-call cost stays bounded.

**Filler skipping.** Tables of contents, page numbers, footers, bibliography entries, and other non-knowledge content surface in extracted text. The LLM is told to skip these (they produce no `new_knowledge` item). The chunk is recorded as "processed but no new knowledge" so progress accounting stays honest.

**File authoritativeness.** Mirroring `domain_authoritativeness`, the config maps filename patterns (glob) and/or path prefixes to trust scalars in `[0, 1]`:

```jsonc
"file_authoritativeness": {
  "acme-handbook-*.pdf":         0.9,
  "/policies/":                  0.8,
  "*.draft.pdf":                 0.3
}
```

Resolution: longest-match wins (path prefix beats bare glob if both match); falls back to `default_file_authoritativeness` (default `0.5`). Effective trust for a document write becomes `max(role_authoritativeness, file_authoritativeness)` — same `max` rule as URL ingest, with file replacing domain.

**Why glob + path, not just filename.** A deployment that organizes uploads under `data/uploads/policies/` and `data/uploads/drafts/` should be able to declare trust by folder. A filename-only map forces every individual file to be listed. Path prefixes give bulk-level policy; globs handle name-pattern conventions ("anything labelled `*.draft.pdf` is low-trust").

**Streaming UX.** Long documents take time. Coco streams progress to stdout as each batch completes (`processed pages 4-6 — saved 1 packet, updated 2`) so the user sees forward motion rather than a multi-minute hang. The streamed reply also doubles as the trace narrative — at the end of the turn, the user has a paragraph summary of what changed.

---

## Identity & roles

Coco runs in one of two startup modes:

- **Anonymous mode** — no login. The user has no name, no email, and an implicit role of `anonymous`. Coco still converses and retrieves, but the role's authoritativeness is 0 — anything an anonymous user contributes is treated as the lowest-trust source.
- **Authenticated mode** — the user signs in via SSO before the first turn. Coco supports two provider families: **Microsoft Entra** for corporate login and **Google** (extensible to other social IdPs) for public login. After a successful login, Coco has a `name`, `email`, and a `role`.

**Which providers are available is configuration-driven.** The list of providers, their client/tenant IDs, scopes, redirect URIs, and the prompt-vs-default behaviour all live in `config.json` under an `auth` block. A deployment that only allows corporate login lists Entra alone; a personal install lists only Google; a hybrid deployment lists both and lets the user choose at the login prompt. Anonymous can be presented as a third option or disabled entirely.

**Role resolution depends on the provider:**

- **Microsoft Entra** — the role is read from the ID-token claims (Entra App Roles or group membership), so the corporate IdP remains the source of truth. Coco does not maintain its own copy of the corporate role table.
- **Google (and any other provider that does not surface roles)** — the role is looked up in a configuration-file map of `email → role`. Email matching is case-insensitive. Emails not in the map fall back to `auth.default_role` (default `user`).

**The role spectrum** (descending authoritativeness):

| Role | Authoritativeness | Intent |
|---|---|---|
| `admin` | 1.0 | Full control — including future destructive ops (force-rewrite content, override conflicts) |
| `author` | 0.8 | Reads and writes packets; the default for trusted contributors |
| `viewer` | 0.5 | Reads packets and converses; cannot mutate long-term memory |
| `user` | 0.3 | Reads, converses, may write to the scratchpad but cannot promote to packets |
| `anonymous` | 0.0 | Read-only; anything contributed carries the lowest trust signal |

**Why "authoritativeness".** The scalar measures *how trustworthy knowledge originating from this role is*. It's not about what the user can do (capabilities cover that) — it's about how much weight the system should give to a fact this user introduces. When two sources disagree, the more authoritative one wins; when several packets compete in retrieval ranking, the more authoritative ones surface first.

**Role-to-capability mapping.** Alongside the scalar power, each role carries a set of named *capabilities* — the functionalities and agentic tools (skills) the role is permitted to invoke. The mapping lives in `config.json` so a deployment can tighten or relax the surface without code changes. A capability not listed for a role is implicitly denied.

Capabilities fall into three families:

- **Memory operations** — `read_packets`, `write_scratchpad`, `promote_scratchpad`, `create_packet`, `integrate_packet`, `override_conflict`.
- **Skills** (named by skill id) — `skill.fetch_url` today; `skill.<future_skill>` as the layer grows. Each skill registers its required capability so the agent can check before invocation.
- **Administrative** — `delete_packet`, `force_rewrite`.

Default capability map (overridable in `config.auth.role_capabilities`):

| Role | Capabilities |
|---|---|
| `admin` | all of the above (including `delete_packet`, `force_rewrite`, `override_conflict`) |
| `author` | `read_packets`, `write_scratchpad`, `promote_scratchpad`, `create_packet`, `integrate_packet`, `skill.fetch_url` |
| `viewer` | `read_packets` |
| `user` | `read_packets`, `write_scratchpad`, `skill.fetch_url` |
| `anonymous` | `read_packets` |

**Why two layers (capabilities + authoritativeness), not one.** Some gates are sharp (delete a packet? override a conflict? call a skill?) and want a binary yes/no — capabilities. Others are smooth (which source wins a contradiction? rank a more-trusted packet higher in retrieval?) and want a scalar — `role_authoritativeness`. Putting both on the role keeps the model honest: hard gates don't accidentally become threshold logic, soft policies don't accidentally collapse into binary cliffs. The two layers are checked independently, in this order:

1. **Capability check** at the tool/operation call site. Denied → operation does not run; Coco surfaces a brief "I'm not able to do that for your account" hint; the denial is traced.
2. **Authoritativeness-weighted decision** (where applicable) — drives conflict resolution during integrate-on-write, biases retrieval ranking toward higher-trust packets, sets the initial trust score of newly stored knowledge. Only reached if the capability check passed.

**Capability source — same as the role.** The Entra branch reads the role string from token claims and then looks up its capability set in the config map (keyed by role string, not by provider). Deployments that want Entra to also override per-user capability sets directly can extend `parse_role_from_entra_claims` to union additional capabilities in from a custom claim — the default keeps a single source of truth at the deployment level.

**Why a spectrum, not boolean.** A bare "is this user allowed?" check forces every gating decision into a binary that always errs on one side. A scalar authoritativeness lets later code phrase soft policies ("higher-trust source wins a contradiction", "weight retrieved content by writer authoritativeness") via thresholds, without re-tagging every action with a discrete capability set.

**Where identity lives.** The acquired identity is attached to the `Session` once at startup and propagated to Langfuse traces (`user_id` = email-or-name, `role` and `role_authoritativeness` as metadata). Identity also rides through to the *packets a user writes* — every packet records the source(s) of its knowledge (see next section), so per-packet writer attribution is no longer deferred: it is the substrate for conflict resolution and trust-weighted retrieval.

**Identity in the agent's context — the user's name (and profile) is known from turn one.** As soon as login completes, the identity's `name` is spliced into Coco's main-reply system prompt at every turn ("You are Coco, a self-learning conversational assistant for **Shishir Choudhary**…"). When available, `email` and `role` are surfaced alongside the name so Coco has a self-contained picture of who she is talking to. This means:

- **No re-introductions.** After a fresh SSO login Coco can greet the user by name in the very first turn — she doesn't need the user to type "hi, I'm Shishir" for her to know. `banner_welcome(identity.name)` at startup and the system-prompt splice both use the same source (`Session.user.name`).
- **Self-referential retrieval works out of the box.** Packets whose facets or entities are keyed on the user's own name ("Shishir's family members", entity `"shishir"`) become naturally in-scope: the entity is *already* present in the agent's context each turn, so the small LM's novelty check and the retrieval channels line up with what Coco actually knows about the current speaker.
- **Anonymous mode stays honest.** In anonymous mode `identity.name` is the literal string `"anonymous"`; the system prompt phrasing degrades gracefully ("…assistant for the user") rather than inventing a name Coco doesn't have. Coco is instructed not to *claim* to know an anonymous user's identity.
- **Profile info beyond the name.** `email` is not surfaced into the reply prompt by default (it's stored on `Session.user` and appears in provenance / traces) — the design assumes emails don't add value to conversational reply framing. Deployments that want richer context (role label, team, department claim from Entra) can extend the system-prompt splice via a small `identity_context_block(Identity) -> str` helper; the default renders just the display name.

Design intent: the moment a user signs in, Coco should treat their name (and the fact of their being logged in) as ambient knowledge — the same way a colleague who just shook your hand doesn't need to keep asking your name mid-conversation.

### Local admin mode (developer / testing only)

Coco supports a **`--admin` command-line flag** that bypasses SSO entirely and drops the session straight into a synthesized `admin` identity. This is a **developer escape hatch** for quick local iteration on prompts, thresholds, and the write path — no IdP round-trip, no email → role lookup, full capabilities from turn one.

**Why it exists.** Tuning Coco is empirical: try a prompt change, restart, watch what happens; try a threshold, restart, watch again. Requiring an interactive SSO login on every restart is the wrong friction. `--admin` collapses "start Coco, do the thing" to a single command; conflict prompts fire, `create_packet` / `integrate_packet` / `override_conflict` all run without capability denials, and traces still capture everything so the tuning session stays inspectable.

**Why it is safe by default.** A CLI flag that bypasses auth would be an obvious backdoor around the entire trust model if it worked in every deployment. So the design forces three independent gates:

1. **Config-gated.** `auth.allow_cli_admin` defaults to `false`. Passing `--admin` when the config disallows it aborts startup with an error before any turn runs. Production configs never set this to `true`; local development configs may.
2. **Explicit flag only.** There is no environment variable, no config option, and no "default admin" fallback that flips a normal startup into admin mode. The user has to type `--admin` on the command line every time.
3. **Honest provenance.** The synthesized identity carries `provider="cli_admin"` and `email=None`. Every packet a `--admin` session writes records `speaker_email=None`, `speaker_role="admin"`, and `provider="cli_admin"` in its `PacketSource`. Deployments that don't trust CLI-admin provenance can filter on that string when curating packets.

**Production reachability of admin capabilities.** In production, admin capabilities are reachable **only** through a real SSO login that resolves the current user to `role == "admin"` via the provider's claims (Entra) or the email → role map (Google/other). The CLI path is closed by config.

**Visual signalling — the user must not lose track of what mode they are in.** Local admin mode confers full write/delete power without a password; if the user forgets the mode is on, they might make mutations they'd never make in a normal session. So every user-facing surface makes the mode unmissable:

- **Startup warning banner (unmissable).** A bright red/yellow warning block is printed before the first turn — bordered, multi-line, and impossible to skim past. Wording explicitly says the session is unauthenticated, that this is unadvisable outside local dev, and that it must not be used in production. On non-TTY stdout the same wording renders as plain ASCII with the same content.
- **Per-turn indicator.** The `You:` prompt is prefixed with a bold red `[ADMIN]` badge, and the `Coco:` reply label is suffixed with a dim red `(admin mode)` marker. The user cannot type a message without seeing the badge on the same line — this survives long conversations where the startup banner has scrolled off.
- **Session summary line.** The goodbye banner at session end reprints a compact `local admin mode — session was unauthenticated` reminder so the user leaves the CLI aware of the mode they just ran in.
- **Trace metadata.** The Langfuse `session_context` records `admin_mode=true`, `provider="cli_admin"`, and `unauthenticated=true`. Post-hoc review can filter to admin-mode sessions and treat their trust claims accordingly.

**Where this does not weaken the trust model.** Packets written in admin mode still get real `PacketSource` entries with `role_authoritativeness = 1.0`. That's deliberate: an admin's word carries admin trust regardless of how the admin was authenticated. What changes is only *how* the admin identity was acquired — via a CLI flag rather than an IdP round-trip — which is honestly recorded on `provider="cli_admin"`. The scalar trust and capabilities are the same as any other admin; the audit trail simply says "this admin came in through the local dev door."

---

## Knowledge source provenance & effective authoritativeness

Every packet records *where its knowledge came from*. This is what makes role authoritativeness actually load-bearing — without provenance, the trust scalar has nothing to attach to.

**Three source types** are tracked per write event into a packet:

- **URL source** — when the knowledge came from URL ingestion. Carries the absolute URL (after redirects) and the *domain authoritativeness* (see below).
- **Conversation source** — when the knowledge came from chat. Carries the speaker's `name`, `email`, `role`, and the role's authoritativeness *at the time of writing*.
- **Document source** — when the knowledge came from an uploaded file (PDF / DOCX / PPTX / text). Carries the `filename`, the detected `document_type` (`word_processing` or `presentation`), the `page_number` and `paragraph_index` of the chunk that produced this write, and the resolved *file authoritativeness*.

Sources accumulate on the packet's `sources` list (append-only). One packet that started as a Wikipedia ingestion and later gets corroborated by a chat conversation AND a paragraph from an uploaded PDF will carry all three — none is overwritten by the others.

**Domain authoritativeness.** Web sources are not equal. Wikipedia, an internal company knowledge base, and a random forum post should not be trusted identically just because they all came through `fetch_url`. The deployment config maps domain (or domain + path prefix) patterns to trust scalars in `[0, 1]`:

```jsonc
"domain_authoritativeness": {
  "en.wikipedia.org":           1.0,
  "docs.python.org":            1.0,
  "internal.acme.com/handbook": 0.9,    // path-prefix match — more specific wins
  "internal.acme.com":          0.7,    // base for the rest of the corporate site
  "medium.com":                 0.4,
  "reddit.com":                 0.2
}
```

Resolution is longest-prefix match against the host + path of the URL. URLs that match no pattern fall back to `default_domain_authoritativeness` (default `0.5` — "neutral, unknown").

**Effective authoritativeness of a write.** When new content lands in a packet, the system computes:

```
effective_authoritativeness = max(role_authoritativeness, source_trust)

where source_trust is:
  - domain_authoritativeness   if PacketSource.type == "url"
  - file_authoritativeness     if PacketSource.type == "document"
  - 0                          if PacketSource.type == "conversation"
```

The `max` is deliberate. The canonical URL case: an `author` (role auth `0.8`) ingests Wikipedia (domain auth `1.0`) → effective `1.0`. The same shape applies for documents: an `author` uploading `acme-handbook-v3.pdf` (file auth `0.9` per the config) writes content at trust `0.9`. Conversely, an `admin` (role auth `1.0`) speaking from memory in chat carries trust `1.0` even though there is no URL or file — their role *is* the source. The source-trust term only ever *raises* the trust beyond the role's; it never drags it down.

For conversation-only writes there is no domain or file term, so `effective_authoritativeness = role_authoritativeness` plainly.

**Per-packet aggregate trust.** A packet's overall authoritativeness is the *maximum* effective trust across all its sources:

```
packet.authoritativeness = max(source.effective_authoritativeness for source in packet.sources)
```

The max (not average, not last-write-wins) is again deliberate: once a packet has been touched by a high-trust source, the *fact* is corroborated; later low-trust additions don't dilute it. They are recorded for transparency but don't lower the bar.

## Conflict resolution by authoritativeness

The original design (v1, early v2) handled write conflicts by pausing the turn and asking the user. That worked for a single-user personal assistant; with multiple users at different trust levels it becomes wrong — a `viewer` shouldn't be able to overwrite an `admin`-authored fact just because they happened to be in the chair.

**The new conflict rule, applied inside integrate-on-write:**

1. Let `new_eff` = effective authoritativeness of the incoming content (computed from the writer's role + the source domain, if any).
2. Let `existing` = `packet.authoritativeness` (the max across what's already in the packet).
3. The LLM doing the merge is told both numbers. It is instructed to:
   - For conflicting facts: **the higher-trust source wins.** If `new_eff > existing` → the new claim replaces the old; if `new_eff < existing` → the existing claim is preserved and the new one is recorded as a less-trusted alternate ("Source X says Y, though more authoritative sources say Z"); if equal → both are surfaced and the user is prompted (only if they have the `override_conflict` capability).
   - For non-conflicting facts: merged as before.
4. After the merge, `packet.authoritativeness = max(existing, new_eff)`.

This shifts conflict resolution from a UX-blocking dialog into an automatic policy that scales to many users and many sources, while still preserving an escape hatch for genuinely equal-trust contradictions.

**Retrieval bias by authoritativeness.** The 3-channel RRF score gets an additional small bias `+ h(packet.authoritativeness)`, paralleling the strength bias. High-trust packets surface earlier when multiple packets compete on otherwise similar matches. The bias scale is configurable; the default is small enough that a sharp semantic match still wins over a weakly-relevant high-trust packet — the bias breaks ties, not arguments.

---

## Skills layer

Coco can call skills (tools) just like any agent. Packets may reference skills in their content ("for upcoming birthdays, query the calendar skill"). When a packet referencing a skill is loaded, Coco has the option to invoke it. No special integration beyond convention.

URL ingestion is implemented as the first such skill: `fetch_url` returns markdown + an image manifest. Unlike user-callable skills, this one is invoked automatically by `chat_turn` whenever the streaming extractor sets `is_ingest_request: true`.

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
| URL ingestion trigger | Explicit verb + URL, detected by the streaming small LM | Auto-fetching every URL the user pastes is invasive and expensive; intent-gated is the conversational equivalent of "go read this for me" |
| URL ingestion write path | Reuses §"turn loop" step 6 — integrate vs scratchpad vs new packet | One source URL can enrich an existing packet rather than fragmenting into a parallel "page" packet; consistent semantics with conversation-driven learning |
| URL ingestion reply | Brief acknowledgement + 2–3 takeaways in the same turn | Mirrors a person summarizing what they just read; gives the user visible confirmation without showing the full extracted markdown |
| URL ingestion provenance | `sources: list[PacketSource]` per packet (URL entries carry the resolved domain authoritativeness); re-ingest refetches and routes to the matching packet | Pages change over time; refetching keeps memory current. Tracking the URL inside structured sources (rather than a plain string list) lets the same field also carry trust info needed for conflict resolution and retrieval bias |
| Image admission policy | LLM-decided at write time, with the image bytes themselves in the prompt as multimodal blocks | Alt text is unreliable. Letting the reply LLM see candidate images directly produces sharply better keep/drop decisions for marginal storage cost |
| Image storage location | Separate `packet.images: list[PacketImage]` on the packet, referenced from `content.full` via `coco-img:<id>` markdown URIs | Inline base64 destroys the text channel — integrate-on-write prompts become huge, unreadable, expensive. Splitting bytes from references keeps text light and lets images load as proper multimodal blocks |
| Image rendering into LLM context | When a packet loads at the `full` slice, its images are attached to the next LLM call as `image` content blocks; lower slices skip images | `gist`/`summary` should remain cheap; `full` already signals "load everything." Slice-gated image loading keeps the strength-band economy honest |
| Image size cap | Downscale to `image_max_dim`; drop if post-downscale > `image_max_bytes` | Bound per-packet storage; avoids absurd 10MB hero photos eating slice budget without information gain |
| Fetch backend | httpx + readability-lxml + markdownify; no JS rendering in v1 | Most knowledge-bearing pages serve usable HTML; a Playwright fallback is deferred work |
| Ingest placement in turn | Skill called inside `_chat_turn_inner` before main_reply | The reply LLM needs the fetched content as context to summarize and emit `new_knowledge`; pre-fetching in streaming would have to predict intent mid-typing |
| Document ingestion as a separate skill | New `skill.upload_document` capability; trigger detected by the streaming small LM ("read this file", "upload this", "add this document") with a path | Files are a fundamentally different surface from URLs (local read, not network; possible large size; needs format-specific parsers); a separate capability and code path keeps both pipelines clean |
| PDF document-type detection | Small LM classifies first ~3 pages as `word_processing` vs `presentation`; result locks chunking strategy for the rest of the document | A prose paragraph and a presentation slide chunk *very* differently; one rule for both gives bad packets either way. Running once up-front is cheap; native DOCX/PPTX skip the classifier entirely |
| Document chunking granularity | WP → paragraph; presentation → slide | Each chunk should be the smallest *self-contained* unit of knowledge for that format. Paragraphs are the natural prose unit; slides are the natural presentation unit |
| Document write loop | Stream chunks in batches (~10) to the main LLM; each chunk routes independently through the write-path | One LLM call per paragraph would be too expensive at 100+ paragraphs; one call for the whole document loses per-paragraph routing. Batching is the cost/granularity midpoint |
| Streaming progress UX | Inline progress chunks ("processed pages 4-6 — saved 1, updated 2") streamed as the doc is processed | Long documents take time. The user needs visible forward motion; the streamed narrative also doubles as a self-summarizing audit log of what changed |
| Document provenance | New `PacketSource` type `"document"` with `filename`, `document_type`, `page_number`, `paragraph_index`, `file_authoritativeness` | One uploaded PDF can contribute to many packets; per-chunk coordinates make the audit trail honest about *which* paragraph in *which* file backed a fact |
| File authoritativeness | Config-driven map of filename glob / path prefix → trust scalar in `[0, 1]`; longest-match wins | Matches the domain-authoritativeness pattern. Path prefixes let policy live at folder granularity (`/policies/` is high-trust); globs handle name conventions (`*.draft.pdf` is low-trust) |
| Filler skipping | LLM is told to drop chunks that are tables of contents, page numbers, footers, bibliography lines (no `new_knowledge` item produced) | Extracted text from PDFs is full of structural noise that has no place in long-term memory; the LLM is the cheapest classifier we already have in the loop |
| Authentication shape | Two startup modes (anonymous, authenticated) + pluggable SSO via config | Personal installs run anonymous; deployments with multiple users get IdP-grade login without invading the core memory model |
| SSO providers | Microsoft Entra (corporate) + Google (public/social); list controlled by `config.json` | Matches the two real audiences — corporate teams and personal/public users — without baking either into the core. Other social IdPs follow the Google pattern |
| Role source | Entra: ID-token claims (App Roles / groups); other providers: email→role config map | Mirrors how each provider expects to be the source of truth. Entra owns corporate role tables; Google does not surface roles, so the config fills the gap |
| Role representation | Scalar `role_authoritativeness` `0.0–1.0` (admin 1.0, author 0.8, viewer 0.5, user 0.3, anonymous 0.0) instead of capability flags | Lets conflict resolution, retrieval ranking, and stored-knowledge trust phrase policies as continuous comparisons; one role can be raised or lowered without re-plumbing every check |
| Naming: "authoritativeness" not "power" | Names the *use*: how much trust knowledge from this role carries | "Power" suggested permissions; authoritativeness pins the scalar to the actual semantic — source trust |
| Anonymous permitted | Anonymous role with authoritativeness 0.0; can be offered alongside SSO or used as the sole mode | Some installs want fully open conversational use; trust-weighted decisions still degrade anonymous contributions automatically |
| Where identity lives | On `Session.user` + propagated to Langfuse trace metadata + recorded on every packet write as a `PacketSource` | Identity is session-scoped at runtime *and* attribution-scoped at the packet level — per-write provenance is now first-class, not deferred |
| User name in the agent's context | On login, `Identity.name` (from SSO claims — display name for Entra / Google) is spliced into Coco's main-reply system prompt every turn; anonymous mode falls back to `"the user"` | Coco should know who she is talking to from turn 1 — greet by name, resolve self-referential packets (`entities: ["shishir"]`), avoid asking "and you are…?" the user just answered via SSO. Reads name from `Session.user.name`, not from a config string, so multi-user deployments work correctly |
| Grounded-reply policy | Substantive answers must be *anchored* by at least one loaded packet fact. General knowledge is permitted only as connective tissue to reason FROM a packet fact toward the answer — never as the standalone source. When no loaded packet is relevant to the question, reply is exactly "I do not know about this." (optionally followed by one line offering to learn) | Coco's value is her accumulated memory, not general LLM competence. Anchoring every answer in a packet gives the user (a) legibility about what Coco actually remembers, (b) preserved write-path pressure so unanswered questions become new packets, and (c) an auditable provenance chain — the packet is the substrate, general knowledge is only the bridge. Allowing packet-anchored reasoning keeps Coco genuinely helpful without collapsing into base-model chat |
| Refusal phrase is fixed, not paraphrased | The exact string "I do not know about this." — not "I don't have that in memory", not "I'm not sure", not "let me look that up" | Consistency lets the user recognize the refusal instantly and reach for the "tell me / share URL / upload file" affordance. Variable phrasings drift toward hedging, which drifts toward guessing |
| Grounded-reply exceptions carve-out | Small closed list: user identity (from login), Coco's self-description, conversational niceties, introspection over loaded state, ingest/upload turns, clarification questions | Zero exceptions makes Coco unusable (can't even say hi). A closed list keeps the carve-outs auditable and prevents drift toward "helpful world knowledge" |
| Refusal turns still run the write path | Even when Coco refuses to answer, any substantive user content in the same turn flows through the scratchpad / packet-write path | Refusing to *answer* isn't the same as refusing to *learn*. Every unanswered question is an opportunity to add a packet so next time Coco can answer |
| Hard vs soft gates | Two layers per role: a capability set (binary checks at call sites) + `role_authoritativeness` (scalar for conflict resolution, retrieval bias, trust accounting) | Some decisions are binary (delete? call skill?), others are smooth (which source wins a contradiction? trust of stored knowledge?); keeping both prevents one concept from bending in both directions |
| Per-packet source provenance | Every write into a packet appends a `PacketSource` record (type: URL or conversation; identity; domain auth or role auth) | Without provenance the trust scalar has nothing to attach to. Sources accumulate so a packet enriched by multiple writers retains the full history |
| Effective authoritativeness of a write | `max(role_authoritativeness, domain_authoritativeness)` | The fact is backed by whichever source is more trustworthy — the person who cited it or the site they cited. Captures the example: author (0.8) + Wikipedia (1.0) → write trust 1.0 |
| Domain authoritativeness config | Map of domain (or domain + path-prefix) → trust scalar; longest-prefix wins | A deployment can declare its own trusted sites without code changes. Path prefixes let `internal.acme.com/handbook` be more trusted than the rest of `internal.acme.com` |
| Packet aggregate trust | `packet.authoritativeness = max(source.effective_authoritativeness)` over all sources | Once a packet has been corroborated by a high-trust source, later low-trust additions don't dilute the fact — they're recorded for transparency but don't lower the bar |
| Conflict resolution: trust-driven not user-prompted | The LLM is told both trust scores; higher-trust source wins; user is asked only when scores tie *and* they hold `override_conflict` | Multi-user systems can't block on UX for every contradiction; trust gives a deterministic rule that scales |
| Retrieval bias by authoritativeness | Add a small `h(packet.authoritativeness)` to the RRF final score, parallel to the strength bias | Trust breaks ties between equally-relevant packets; default scale is small so sharp semantic matches still win |
| Capability source | Default map in `config.auth.role_capabilities` keyed by role string | One source of truth for the deployment; Entra claims still resolve the role string, and the role string then maps to capabilities. Future per-user capability overrides via claims remain optional |
| Default capability tiering | admin = all; author/user = write paths + `skill.fetch_url`; viewer/anonymous = read-only | Mirrors the trust spectrum; "viewer" is a strict read-only conversational seat, "user" is the default writer for personal installs |
| Local admin mode (CLI flag) | `--admin` at process start synthesizes a full-trust admin `Identity` without SSO; gated by `auth.allow_cli_admin` (default `false`); visually flagged by a startup warning banner and a per-turn `[ADMIN]` prompt badge; every packet's `PacketSource` records `provider="cli_admin"` | Tuning prompts and thresholds needs many quick startups where an SSO round-trip is wrong friction; but a flag that bypasses auth cannot be reachable in production. Config-gate + explicit CLI flag + persistent visual highlighting keep the escape hatch honest — a developer can move fast locally, but the mode is unmissable and provenance is preserved on every write |

---

## Open / deferred items

- **Image-as-retrieval-channel.** Images now live as first-class `PacketImage` records (un-deferred via URL ingestion). What remains deferred: whether image content also contributes its own vector or entity-like handle for retrieval — currently images are only visible *after* a packet is loaded by topic/entity match.
- **Orphan-image garbage collection.** Integrate-on-write may drop a `coco-img:<id>` reference from `content.full` while leaving the `PacketImage` record behind. Tolerated for now (no broken-render risk; the unused bytes just sit on disk). A periodic GC pass that removes images not referenced anywhere in `content.full` is future work.
- **JS-rendered pages.** v1 fetch is static-HTML only. Playwright (or similar) fallback for SPA / Notion / JS-only pages is future work.
- **Multi-page sites.** Coco ingests exactly the page the URL points to. Following outbound links, paginated articles, or whole-site crawls is out of scope for v1.
- **Non-HTML URL resources.** Videos, audio, and binary URL downloads are still rejected at fetch time. (PDF / DOCX / PPTX are no longer URL-fetched — they go through the document-upload path, see "Document ingestion".) Audio / video transcription remains future work.
- **Document images and tables.** v2 document ingestion handles text only. Embedded images, figures, charts, and tables in PDF/DOCX/PPTX are extracted as text where possible (alt text, OCR-free table headers) and skipped otherwise. Multimodal extraction is future work — the existing `PacketImage` machinery from URL ingest is the natural target to wire in.
- **OCR for image-only PDFs.** Scanned PDFs without an embedded text layer surface as empty pages and are skipped. A Tesseract (or hosted OCR) fallback is future work.
- **Strength event compaction.** The `strength_events` log grows monotonically. Periodic compaction (collapse old events into a decayed scalar + reset event log) is future work.
- **Initial bootstrap.** Coco starts empty: no packets, no scratchpad. Basic facts (user's name, date) learned in the first conversation like anything else.
- **Skills layer details.** How exactly a packet references a skill, and whether Coco autonomously invokes vs. proposes-and-asks, is left to implementation. URL ingestion is the first agent-auto-invoked skill; user-callable skills follow the same pattern.
- **Embedding-based entity aliasing.** Currently entities match by lowercased text only. "Alka" and "my wife" won't co-resolve. v2 future work.
- **Facet pruning.** Nothing currently caps the number of facets a packet can accumulate. If a packet drifts to 20+ facets and they get noisy, consider periodic facet consolidation via LLM call.
- **Additional SSO providers.** v2 ships Microsoft Entra + Google. GitHub, Okta, Auth0, etc. plug into the same `auth.providers` config slot via the Google-style "email → role from config" pattern.
- **Argument-level capability gating.** Today capabilities are coarse (`skill.fetch_url` is on or off for a role). Argument-level policies — e.g. an allowlist of domains for `anonymous`, or a max-image-budget tied to `role_authoritativeness` — are deferred.
- **Per-source strength decay.** Authoritativeness on a packet is currently a max-aggregate over all sources, with no time decay. A source ingested two years ago and never re-corroborated still pegs the packet at full trust. A decayed-authoritativeness pass (mirroring `strength` decay) is future work.
- **Authoritativeness-weighted retrieval.** Wired in v2 as a small additive bias. Tuning the scale and exploring more sophisticated blends (e.g. multiplicative on partial-match packets only) is left for empirical iteration.
- **Source authoritativeness for non-URL non-conversation inputs.** v2 covers URL ingest and chat. Future inputs (PDF uploads, file watchers, IMAP) will need to declare how their source trust is resolved — config-mapped by file path, by sender email, etc.
- **Capability auditing dashboard.** Denied capability checks are logged to Langfuse traces, but there is no aggregate view of "which roles bump into which gates." A periodic report on denial counts per (role, capability) pair is future work.
- **Per-user capability overrides via claims.** Default policy reads capabilities from the role string in config. A future extension can let Entra carry a `scopes` (or custom) claim that unions extra capabilities onto the role's baseline for specific users without changing the deployment config.
- **Runtime / embedding model / hybrid-search implementation.** All deferred to implementation planning.
