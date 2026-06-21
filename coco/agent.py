"""Turn loop orchestration, LLM calls, write-path branching.

Streaming is the single retrieval and topic-classification path. `on_text_event`
runs for both partial and submit events. `chat_turn` is purely the main reply +
write-path; it does no retrieval or topic resolution.
"""
import asyncio
import sys
from datetime import datetime, timezone
import numpy as np

from . import streaming, tracing, ui
from .llm import anthropic_client
from .config import load_config
from .embeddings import embed, get_model
from .extraction import extract_partial
from .memory import Packet, ScratchpadEntry, PacketStore, Scratchpad, SessionCounter
from .prompts import (
    INTEGRATE_PROMPT,
    NEW_PACKET_PROMPT,
    build_system_prompt,
    build_user_message,
    parse_integration_response,
    parse_new_packet_response,
    parse_coco_response,
)
from .retrieval import (
    best_packet_facet_match,
    best_scratchpad_match,
    rrf_packet_search,
)
from .session import Session
from .strength import compute_strength, slice_for_strength, strength_bias


# -----------------------------------------------------------------------------
# Streaming-aware Claude call
# -----------------------------------------------------------------------------

_REPLY_START = "<reply>"
_REPLY_END = "</reply>"


def _flush_streaming(
    full_text: str, last_emitted: int, state: str
) -> tuple[int, str]:
    """Print any newly-visible <reply>...</reply> content. Returns (last_emitted, state)."""
    if state == "after":
        return last_emitted, state

    if state == "before":
        start = full_text.find(_REPLY_START)
        if start < 0:
            return last_emitted, state
        content_start = start + len(_REPLY_START)
        if content_start < len(full_text) and full_text[content_start] == "\n":
            content_start += 1
        last_emitted = content_start
        state = "inside"

    if state == "inside":
        end = full_text.find(_REPLY_END, last_emitted)
        if end >= 0:
            chunk = full_text[last_emitted:end]
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            last_emitted = end + len(_REPLY_END)
            state = "after"
        else:
            chunk = full_text[last_emitted:]
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            last_emitted = len(full_text)

    return last_emitted, state


async def _claude_call(
    system_prompt: str,
    user_message: str,
    model: str,
    span_name: str = "claude_call",
    stream_to_stdout: bool = False,
    max_tokens: int = 4096,
) -> str:
    """Direct Anthropic SDK call via messages.stream. Returns full text."""
    with tracing.observation(
        span_name,
        as_type="generation",
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    ) as gen:
        text = ""
        state = "before"
        last_emitted = 0

        client = anthropic_client()
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for delta in stream.text_stream:
                text += delta
                if stream_to_stdout and delta:
                    last_emitted, state = _flush_streaming(text, last_emitted, state)

        if stream_to_stdout:
            last_emitted, state = _flush_streaming(text, last_emitted, state)
            if state == "before":
                sys.stdout.write(text)
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

        tracing.update(gen, output=text)
    return text


# -----------------------------------------------------------------------------
# Retrieval (only called from on_text_event)
# -----------------------------------------------------------------------------

def _retrieve_packets(
    query_text: str,
    query_vec,
    session: Session,
    store: PacketStore,
    config: dict,
    span_name: str = "streaming_retrieval",
) -> list[dict]:
    """Returns list of {id, slice, topic} dicts for newly loaded packets."""
    candidates = store.all()
    with tracing.observation(
        span_name,
        as_type="span",
        input={"query": query_text, "candidate_count": len(candidates)},
    ) as span:
        if not candidates:
            tracing.update(span, output={"loaded": []})
            return []

        ranked = rrf_packet_search(
            query_text,
            query_vec,
            candidates,
            k=config["hybrid_search_k"],
            top_n=2,
            debug=config.get("debug_print_streaming", False),
            cosine_floor=config.get("cosine_channel_floor", 0.0),
        )
        now = datetime.now(timezone.utc)
        newly_loaded: list[dict] = []
        considered: list[dict] = []

        for pkt, base_score in ranked[:5]:
            strength = compute_strength(
                pkt.strength_events,
                config["strength_weights"],
                config["strength_half_life_days"],
                now=now,
            )
            final = base_score + strength_bias(strength, config["strength_additive_bias_scale"])

            considered.append({
                "id": pkt.id,
                "topics": [t.text for t in pkt.topics],
                "rrf": round(base_score, 6),
                "strength": round(strength, 3),
                "final": round(final, 6),
            })
            if pkt.id in session.loaded_packets:
                continue
            if final >= config["retrieval_threshold"]:
                slice_type = slice_for_strength(
                    strength,
                    config["band_gist_max"],
                    config["band_summary_max"],
                )
                session.loaded_packets[pkt.id] = {"packet": pkt, "slice": slice_type}
                pkt.record_event("retrieval", config["strength_weights"]["retrieval"])
                store.save(pkt)
                first_topic = pkt.topics[0].text if pkt.topics else ""
                newly_loaded.append({
                    "id": pkt.id,
                    "slice": slice_type,
                    "topic": first_topic,
                    "score": final,
                })
                ui.memory_recall("{gist} {final_score:.2f}/{base_score:.2f}/{strength:.2f}".format( gist= pkt.content.gist or first_topic, final_score=final,base_score=base_score,strength=strength_bias(strength, config["strength_additive_bias_scale"]) ))
                tracing.score(
                    name="retrieval", value=1, comment=f"loaded {pkt.id} ({slice_type})"
                )

        tracing.update(span, output={"loaded": newly_loaded, "considered": considered})
        return newly_loaded


# -----------------------------------------------------------------------------
# Debug state printing
# -----------------------------------------------------------------------------

def _print_state(session: Session, prefix: str = "[state]") -> None:
    entities = _existing_entities_from_session(session)
    print(
        f"\n{prefix} {len(session.loaded_packets)} loaded packet(s)"
        f" | {len(session.topics)} topic(s) | {len(entities)} entity/entities",
        flush=True,
    )
    for pkt_id, item in session.loaded_packets.items():
        pkt = item["packet"]
        topic = pkt.topics[0].text if pkt.topics else "(no topic)"
        gist = pkt.content.gist or pkt.content.summary[:80] or "(empty)"
        print(f"  {pkt_id} [{item['slice']}] \"{topic}\"\n      gist: {gist}", flush=True)
    if session.topics:
        topic_list = ", ".join(f"\"{t['topic_text']}\"" for t in session.topics)
        print(f"  topics: {topic_list}", flush=True)
    if entities:
        ents = ", ".join(entities[:25])
        more = f" (+{len(entities) - 25} more)" if len(entities) > 25 else ""
        print(f"  entities: {ents}{more}", flush=True)
    print(flush=True)


# -----------------------------------------------------------------------------
# The single text-event handler (used for both partial and submit)
# -----------------------------------------------------------------------------

def _existing_entities_from_session(session: Session) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in session.loaded_packets.values():
        for e in item["packet"].entities:
            if e not in seen:
                seen.add(e)
                ordered.append(e)
    return ordered


async def on_text_event(
    text: str,
    session: Session,
    store: PacketStore,
    scratchpad: Scratchpad,
    config: dict,
    event_kind: str,
) -> None:
    """Single retrieval and topic-classification path.

    event_kind is "partial" or "submit"; determines trace name.
    Reads existing session topics + entities, calls extract_partial, guards
    on (is_meaningful AND (has_new_topic OR has_new_entities)), retrieves.
    """
    span_name = f"{event_kind}_event"
    with tracing.observation(
        span_name,
        as_type="span",
        input={"text": text, "kind": event_kind},
        metadata={"session_id": session.id},
    ) as root:
        existing_topics = [t["topic_text"] for t in session.topics]
        existing_entities = _existing_entities_from_session(session)

        result = await extract_partial(
            text, existing_topics, existing_entities, config
        )

        should_retrieve = (
            result["is_meaningful"]
            and (result["has_new_topic"] or result["has_new_entities"])
        )

        tracing.update(root, output={
            "is_meaningful": result["is_meaningful"],
            "has_new_topic": result["has_new_topic"],
            "has_new_entities": result["has_new_entities"],
            "topic_summary": result["topic_summary"],
            "entities": result["entities"],
            "reason": result["reason"],
            "should_retrieve": should_retrieve,
        })

        debug = config.get("debug_print_streaming", True)
        prefix = f"[{event_kind}]"
        word_count = len(text.split())

        if not should_retrieve:
            if debug:
                if not result["is_meaningful"]:
                    print(f"{prefix} {word_count} words → skipped ({result['reason']})", flush=True)
                else:
                    print(f"{prefix} {word_count} words → no new ({result['reason']})", flush=True)
            return

        if result["has_new_topic"] and result["topic_summary"] and result["topic_vec"] is not None:
            session.add_topic(result["topic_summary"], result["topic_vec"])

        newly_loaded: list[dict] = []
        if result["topic_vec"] is not None and result["topic_summary"]:
            # Retrieval uses the small-LM's topic_summary (text + vector),
            # not the raw user text — keeps Channel A BM25 and Channel C
            # entity-bag matching consistent with the way packet facets are
            # phrased ("Shishir's family members" not "tell me about my family").
            newly_loaded = _retrieve_packets(
                result["topic_summary"],
                result["topic_vec"],
                session,
                store,
                config,
                span_name="streaming_retrieval",
            )

        if debug:
            bits: list[str] = []
            if result["has_new_topic"]:
                bits.append(f"new topic \"{result['topic_summary']}\"")
            if result["has_new_entities"]:
                new_ents = [e for e in result["entities"] if e not in existing_entities]
                if new_ents:
                    bits.append(f"new entities {new_ents}")
            change_str = "; ".join(bits) if bits else "(extracted, no novelty flags?)"
            print(f"{prefix} {word_count} words → {change_str}", flush=True)
            for nl in newly_loaded:
                print(
                    f"  loaded {nl['id']} (score {nl['score']:.4f}) [{nl['slice']}] \"{nl['topic']}\"",
                    flush=True,
                )


# -----------------------------------------------------------------------------
# Write path helpers
# -----------------------------------------------------------------------------

async def _create_packet_from_seed(
    seed_topic: str,
    seed_content: str,
    store: PacketStore,
    config: dict,
    session_id: str,
) -> Packet | None:
    prompt = NEW_PACKET_PROMPT.format(seed_topic=seed_topic, seed_content=seed_content)
    raw = await _claude_call(
        system_prompt="You are an expert at extracting clean knowledge from raw conversation excerpts.",
        user_message=prompt,
        model=config["anthropic_model"],
        span_name="new_packet_creation",
    )
    try:
        d = parse_new_packet_response(raw)
    except Exception as e:
        print(f"\n[new-packet parse error: {e}]")
        return None

    facets_texts = d.get("topic_facets") or [seed_topic]
    if not facets_texts:
        facets_texts = [seed_topic]
    facet_pairs = [(t, embed(t, config["embedding_model"])) for t in facets_texts]
    entities = d.get("entities") or []

    pkt = Packet.new(
        topics=facet_pairs,
        entities=entities,
        gist=d.get("gist", ""),
        summary=d.get("summary", ""),
        full=d.get("full", seed_content),
    )
    pkt.source_session_ids.append(session_id)
    pkt.record_event("write", config["strength_weights"]["write"])
    store.add(pkt)
    ui.memory_saved(pkt.content.gist or (pkt.topics[0].text if pkt.topics else ""))
    tracing.score(name="write", value=1, comment=f"created {pkt.id}")
    return pkt


async def integrate_into_packet(
    packet: Packet, new_content: str, store: PacketStore, config: dict
) -> bool:
    facets_str = ", ".join(f'"{t.text}"' for t in packet.topics)
    entities_str = ", ".join(packet.entities[:30])
    prompt = INTEGRATE_PROMPT.format(
        existing_facets=facets_str,
        existing_entities=entities_str,
        existing_content=packet.content.full or packet.content.summary or packet.content.gist,
        new_content=new_content,
    )
    raw = await _claude_call(
        system_prompt="You are an expert at editing and integrating knowledge concisely.",
        user_message=prompt,
        model=config["anthropic_model"],
        span_name="integrate_on_write",
    )
    try:
        integrated = parse_integration_response(raw)
    except Exception as e:
        print(f"\n[integrate parse error: {e}]")
        return False

    if integrated.get("conflict_detected"):
        desc = integrated.get("conflicting_excerpts") or "(no description)"
        topic_name = packet.topics[0].text if packet.topics else packet.id
        print(f"\n[Conflict in packet '{topic_name}']")
        print(f"  Description: {desc}")
        print(f"  New info:    {new_content}")
        choice = input("  Apply update? [y/N]: ").strip().lower()
        if choice != "y":
            print("  Skipped.")
            return False

    packet.content.gist = integrated.get("gist", packet.content.gist)
    packet.content.summary = integrated.get("summary", packet.content.summary)
    packet.content.full = integrated.get("full", packet.content.full)

    new_facet_texts = integrated.get("topic_facets") or []
    for t in new_facet_texts:
        if not t or not t.strip():
            continue
        if any(t.strip().lower() == ex.text.strip().lower() for ex in packet.topics):
            continue
        v = embed(t, config["embedding_model"])
        packet.add_facet_if_new(t, v, config["facet_dedup_threshold"])

    packet.merge_entities(integrated.get("entities") or [])

    packet.record_event("write", config["strength_weights"]["write"])
    store.save(packet)
    ui.memory_updated(packet.content.gist or (packet.topics[0].text if packet.topics else ""))
    tracing.score(name="write", value=1, comment=f"updated {packet.id}")
    return True


# -----------------------------------------------------------------------------
# chat_turn — reply only, no retrieval / topic resolution
# -----------------------------------------------------------------------------

async def chat_turn(
    user_message: str,
    session: Session,
    store: PacketStore,
    scratchpad: Scratchpad,
    config: dict,
    session_n: int,
) -> str:
    with tracing.observation(
        "chat_turn",
        as_type="span",
        input={"user_message": user_message},
        metadata={
            "session_id": session.id,
            "session_n": session_n,
            "turn_index": len(session.turns),
        },
    ) as root:
        reply = await _chat_turn_inner(
            user_message, session, store, scratchpad, config, session_n
        )
        tracing.update(root, output={
            "reply": reply,
            "loaded_packets": list(session.loaded_packets.keys()),
        })
        return reply


async def _chat_turn_inner(
    user_message: str,
    session: Session,
    store: PacketStore,
    scratchpad: Scratchpad,
    config: dict,
    session_n: int,
) -> str:
    session.add_turn("user", user_message)

    system_prompt = build_system_prompt(
        session.loaded_packets_list(), user_name=config["user_name"]
    )
    history = session.turns[:-1][-(config["recency_window"] * 2):]
    framed = build_user_message(history, user_message)

    ui.coco_label()
    raw = await _claude_call(
        system_prompt,
        framed,
        config["anthropic_model"],
        span_name="main_reply",
        stream_to_stdout=True,
    )

    parsed = parse_coco_response(raw)
    reply = parsed["reply"]
    packets_used = parsed["packets_used"]
    new_knowledge = parsed["new_knowledge"]

    # Record use events
    for pid in packets_used:
        item = session.loaded_packets.get(pid)
        if item is not None:
            pkt = item["packet"]
            pkt.record_event("use", config["strength_weights"]["use"])
            store.save(pkt)
            tracing.score(name="use", value=1, comment=f"used {pid}")

    # Write path
    current_topic = None
    topic_vec = None
    if session.current_topic_idx is not None and 0 <= session.current_topic_idx < len(session.topics):
        current_topic = session.topics[session.current_topic_idx]
        topic_vec = np.asarray(current_topic["topic_vector"], dtype=np.float32)

    for nk in new_knowledge:
        content = (nk.get("content") or "").strip()
        if not content:
            continue
        conflicts_with = nk.get("conflicts_with")

        if conflicts_with and conflicts_with in session.loaded_packets:
            target = session.loaded_packets[conflicts_with]["packet"]
            await integrate_into_packet(target, content, store, config)
            continue

        if topic_vec is None:
            continue  # no current topic; can't do facet matching

        all_packets = store.all()
        best_pkt, best_score = best_packet_facet_match(topic_vec, all_packets)
        if best_pkt is not None and best_score >= config["existing_packet_match_threshold"]:
            await integrate_into_packet(best_pkt, content, store, config)
            if best_pkt.id not in session.loaded_packets:
                strength = compute_strength(
                    best_pkt.strength_events,
                    config["strength_weights"],
                    config["strength_half_life_days"],
                )
                slice_type = slice_for_strength(
                    strength, config["band_gist_max"], config["band_summary_max"]
                )
                session.loaded_packets[best_pkt.id] = {"packet": best_pkt, "slice": slice_type}
            continue

        # Scratchpad path
        entries = scratchpad.all()
        match, _ = best_scratchpad_match(
            topic_vec, entries, config["scratchpad_promote_threshold"]
        )
        if match is not None:
            seed_content = "\n\n".join(match.raw_excerpts + [user_message, content])
            new_pkt = await _create_packet_from_seed(
                seed_topic=match.topic,
                seed_content=seed_content,
                store=store,
                config=config,
                session_id=session.id,
            )
            if new_pkt is not None:
                scratchpad.remove(match.id)
                session.loaded_packets[new_pkt.id] = {"packet": new_pkt, "slice": "full"}
                if config.get("debug_print_streaming", False):
                    print(f"[promoted scratchpad entry to packet: {[t.text for t in new_pkt.topics]}]")
        else:
            topic_str = current_topic["topic_text"] if current_topic else "untitled"
            entry = ScratchpadEntry.new(
                topic=topic_str,
                topic_vector=topic_vec,
                excerpt=f"{user_message}\n\nFact: {content}",
                session_id=session.id,
                session_n=session_n,
            )
            scratchpad.add(entry)

    session.add_turn("coco", reply)
    return reply


# -----------------------------------------------------------------------------
# run_session — streaming event loop
# -----------------------------------------------------------------------------

async def run_session():
    config = load_config()
    store = PacketStore(config["data_dir"])
    scratchpad = Scratchpad(config["data_dir"])
    counter = SessionCounter(config["data_dir"])
    session_n = counter.increment()
    scratchpad.prune_old(session_n, config["scratchpad_discard_after_sessions"])

    tracing_on = tracing.init()

    # Pre-load embedding model so the first extraction is not blocked on it.
    get_model(config["embedding_model"])

    session = Session()

    ui.banner_welcome(config.get("user_name", ""))

    debug_state = config.get("debug_print_state", False)
    debug_streaming = config.get("debug_print_streaming", False)
    if debug_state or debug_streaming:
        ui.hint(
            f"developer mode · session {session.id} · "
            f"{len(store.all())} packets, {len(scratchpad.all())} scratchpad · "
            f"tracing {'on' if tracing_on else 'off'}"
        )
    if debug_state:
        _print_state(session, prefix="[state · session start]")

    with tracing.session_context(session.id, config.get("user_name")):
        in_flight: set[asyncio.Task] = set()

        async for event in streaming.input_stream(config):
            if event.kind == "cancel":
                break

            if event.kind == "partial":
                task = asyncio.create_task(
                    on_text_event(event.text, session, store, scratchpad, config, "partial")
                )
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
                continue

            # submit
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
                in_flight.clear()

            try:
                await on_text_event(event.text, session, store, scratchpad, config, "submit")
            except Exception as e:
                print(f"\n[submit-extraction error: {e}]")

            try:
                await chat_turn(event.text, session, store, scratchpad, config, session_n)
            except Exception as e:
                print(f"\n[turn error: {e}]")

            if debug_state:
                _print_state(session, prefix="[state · after turn]")

    if debug_state or debug_streaming:
        ui.hint(
            f"developer mode · session ended · "
            f"{len(session.topics)} topics, {len(session.loaded_packets)} packets loaded"
        )
    ui.banner_goodbye()
    tracing.flush()
