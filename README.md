# Coco — a self-learning conversational knowledge agent

> Coco is a conversational agent that learns from every conversation. As you type, a small language model continuously classifies the topic and entities you're discussing, retrieves the relevant knowledge from long-term memory, and streams a response that uses what it remembers. After each turn, the agent decides what was worth remembering and consolidates it into searchable memory packets.

## What makes Coco different

- **Streaming-first.** Topic classification and packet retrieval happen *while you type*, not after you submit. A small LM (Haiku) extracts topic + entities + novelty flags on each partial; by the time you press Enter, the right memories are already loaded.
- **Multi-entry-point memory.** Each "packet" of knowledge has multiple topic facets and an entity list — so a memory about a person is reachable from their name, their habits, their family, or just the entity mention alone.
- **Three-channel hybrid retrieval.** Reciprocal Rank Fusion over topic BM25, max-cosine across topic facets, and entity-bag BM25. Per-channel zero-score filtering keeps irrelevant packets out of the score.
- **Multi-fidelity content.** Packets store a gist, a summary, and the full content. A dynamic strength score (decaying weighted sum of retrieval / use / write events) controls which slice loads each turn.
- **Real token-level streaming.** Direct Anthropic SDK (`messages.stream`) for the main reply — text appears as it's generated, not in stdio-batched chunks.
- **Langfuse-instrumented.** Every LLM call and retrieval is traced; sessions group all turns of one conversation.
- **End-user UX by default.** Clean banner, colored prompt, brief memory-activity hints. Developer mode (with full per-channel score breakdowns and state dumps) is opt-in via flags.

## Architecture at a glance

```
┌─────────────────────┐      ┌────────────────────────┐      ┌──────────────────────┐
│ Streaming console   │      │ Small LM extraction    │      │ Multi-channel RRF    │
│ (per keystroke)     │ ───► │ (Haiku via direct API) │ ───► │ over packet store    │
│ debounce + N-word   │      │ topic, entities,       │      │ topic BM25 +         │
│ trigger             │      │ novelty flags          │      │ max-cosine +         │
└─────────────────────┘      └────────────────────────┘      │ entity BM25          │
                                                             └──────────────────────┘
                                                                       │
                                                                       ▼
                              ┌────────────────────────┐      ┌──────────────────────┐
                              │ Main reply (Sonnet,    │      │ Loaded packets at    │
                              │ streamed via XML       │ ◄─── │ strength-appropriate │
                              │ <reply> tag)           │      │ slice                │
                              └────────────────────────┘      └──────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Write path:            │
                              │ integrate-on-write OR  │
                              │ scratchpad → promote   │
                              └────────────────────────┘
```

For the conceptual design (the "why"): see [`DESIGN.md`](./DESIGN.md).
For the implementation specification with module reference and Mermaid sequence diagrams (the "how"): see [`TDS.md`](./TDS.md).

## Getting started

### Prerequisites

- Python 3.10 or newer
- An Anthropic API key
- Optional: Langfuse account (cloud or self-hosted) for tracing

### Install

```bash
git clone https://github.com/shishircc/self-learning-knowledge-agent.git
cd self-learning-knowledge-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

First install pulls `anthropic`, `sentence-transformers` (and PyTorch — large download on first run), `rank-bm25`, `prompt_toolkit`, `langfuse`, and `python-dotenv`.

### Configure

Create a `.env` file at the project root:

```ini
ANTHROPIC_API_KEY=sk-ant-...

# Optional — Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

If the Langfuse keys are absent, Coco runs with tracing silently disabled.

Tuning knobs live in [`config.json`](./config.json) — thresholds, RRF parameters, strength weights, model IDs, streaming triggers, and the developer-mode toggles. Defaults are sensible for a personal-scale corpus.

### Run

```bash
python -m coco
```

You'll see a welcome banner, then a `You:` prompt. Type a message and watch the dim `recalling: ...` / `remembered: ...` / `updated: ...` hints between turns as Coco's long-term memory grows. Press Ctrl-D or type `exit` to end the session.

## How it works (quick read)

A **packet** is the atomic unit of long-term memory: a list of topic-facet phrases (each with an embedding), a list of entity strings, and content stored at three fidelity levels (gist, summary, full). Packets live as one JSON file per packet under `data/packets/`.

A **session** holds the in-memory state for one run: the topics encountered so far, the packets currently loaded into context, and the turn history.

A **scratchpad** is the short-term buffer for things you've mentioned but haven't yet earned a packet for. After two semantically near-duplicate mentions (typically across sessions), a scratchpad entry is promoted to a real packet.

**Each turn** is driven by streaming: a small LM is called once per debounced partial (and once at submit) to extract `{is_meaningful, has_new_topic, has_new_entities, topic_summary, entities}` from what you've typed so far. If the result is meaningful AND introduces a new topic or new entities, the agent embeds the topic, adds it to the session, and runs 3-channel RRF against the packet store to pull in relevant memory. The main reply LLM then generates a streamed conversational response using the loaded packets as its knowledge.

After the reply, a write-path decides whether the turn yielded new knowledge — and if so, whether it should be merged into an existing packet, promote a scratchpad entry into a new packet, or seed a fresh scratchpad entry.

Full conceptual rationale is in [`DESIGN.md`](./DESIGN.md). Implementation details (module-by-module, with sequence diagrams) are in [`TDS.md`](./TDS.md).

## Project structure

```
.
├── DESIGN.md                conceptual design
├── TDS.md                   technical design specification
├── pyproject.toml
├── config.json              all runtime tuning knobs
└── coco/
    ├── __main__.py          CLI entrypoint (loads .env, runs the streaming loop)
    ├── agent.py             on_text_event + chat_turn + write path
    ├── streaming.py         keystroke streaming with prompt_toolkit + patch_stdout
    ├── extraction.py        small-LM novelty extraction
    ├── retrieval.py         3-channel RRF, zero-score filtering, debug breakdown
    ├── memory.py            Packet / ScratchpadEntry / TopicFacet schemas + storage
    ├── session.py           in-memory session state
    ├── strength.py          decayed-strength computation + slice bands
    ├── prompts.py           XML-tagged main reply, extraction, integrate, new-packet prompts
    ├── llm.py               lazy AsyncAnthropic client
    ├── embeddings.py        sentence-transformers wrapper
    ├── tracing.py           Langfuse wrapper (no-op when keys are absent)
    ├── ui.py                end-user UX helpers (banners, prompt, memory hints)
    └── config.py            defaults + JSON config loader
```

## Developer mode

For prompt and threshold tuning, set either flag in `config.json` to `true`:

- `debug_print_state` — at session start and after each turn, dump loaded-packet IDs / slices / topics / entities
- `debug_print_streaming` — per partial/submit, print the small-LM extraction outcome and the full 3-channel RRF score breakdown (raw / rank / contribution per channel)

End-user mode (the default) suppresses all of this — only conversation, memory-activity hints, and banners reach the console.

## Contributing

Pull requests welcome. A few suggestions to keep the contribution loop smooth:

1. **Open an issue first** for non-trivial changes — especially anything that touches the architecture in `DESIGN.md` or the contracts in `TDS.md`. Small bug fixes, prompt tweaks, or threshold-tuning experiments don't need this step.
2. **Keep the docs in sync with the code.** If you change a public function signature or a config knob, update `TDS.md` in the same PR. If you change a design decision, update `DESIGN.md`'s decision log.
3. **Mermaid sequence diagrams** in `TDS.md` are picky — avoid semicolons, em-dashes, and parenthesized prefixes inside arrow message text. If a diagram fails to render, that's almost always why.
4. **No new dependencies** without a short justification in the PR description. The current set is intentionally minimal.
5. **Quick sanity check before pushing:**
   ```bash
   python -c "
   import ast, pathlib
   for f in pathlib.Path('coco').glob('*.py'):
       ast.parse(f.read_text())
   print('all modules parse cleanly')
   "
   ```

Issues, ideas, and prompt/threshold experiments are all welcome.

## License

MIT — see [`LICENSE`](./LICENSE) for the full text. In short: use it freely in personal or commercial projects, keep the copyright notice, no warranty.
