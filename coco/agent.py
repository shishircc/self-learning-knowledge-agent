"""Turn loop orchestration, LLM calls, write-path branching.

Streaming is the single retrieval and topic-classification path. `on_text_event`
runs for both partial and submit events. `chat_turn` is purely the main reply +
write-path; it does no retrieval or topic resolution.
"""
import asyncio
import re
import sys
from datetime import datetime, timezone
import numpy as np

from . import auth, documents as documents_module, fetch as fetch_module
from . import streaming, tracing, ui
from .auth import (
    AuthError,
    Identity,
    acquire_identity,
    effective_authoritativeness,
    resolve_domain_authoritativeness,
)
from .llm import anthropic_client
from .config import load_config
from .embeddings import embed, get_model
from .extraction import extract_partial
from .memory import (
    Packet,
    PacketImage,
    PacketSource,
    ScratchpadEntry,
    PacketStore,
    Scratchpad,
    SessionCounter,
)
from .prompts import (
    build_document_batch_user_blocks,
    build_system_prompt,
    build_user_content_blocks,
    parse_integration_response,
    parse_new_packet_response,
    parse_coco_response,
    render_integrate_prompt,
    render_new_packet_prompt,
)
from .retrieval import (
    authoritativeness_bias,
    best_packet_facet_match,
    best_scratchpad_match,
    rank_packet_facet_candidates,
    rank_scratchpad_candidates,
    rrf_packet_search,
)
from .session import Session
from .strength import compute_strength, slice_for_strength, strength_bias


_IMG_PLACEHOLDER_RE = re.compile(r"\[IMG_(\d+)\]")


# -----------------------------------------------------------------------------
# Capability gating
# -----------------------------------------------------------------------------

# Hints are rate-limited to one per turn to avoid spamming the user when many
# new_knowledge items in a single batch all hit the same denial. The set is
# cleared at the start of each chat_turn / on_text_event.
_DENIAL_HINTS_THIS_TURN: set[str] = set()


def _reset_denial_hints() -> None:
    _DENIAL_HINTS_THIS_TURN.clear()


def _capability_denied(writer: Identity, capability: str) -> None:
    """Record the denial on the current trace and surface a hint at most once
    per (turn, capability) pair. Never raises — callers branch on the missing
    capability themselves via `writer.can(...)`.
    """
    tracing.score(
        name="capability_denied", value=1,
        comment=f"{writer.role}:{capability}",
    )
    if capability in _DENIAL_HINTS_THIS_TURN:
        return
    _DENIAL_HINTS_THIS_TURN.add(capability)
    if capability == "skill.fetch_url":
        ui.hint("  I'm not able to read web pages for your account.")
    elif capability in {"create_packet", "integrate_packet", "promote_scratchpad"}:
        ui.hint("  I'm not able to update my long-term memory for your account.")
    elif capability == "write_scratchpad":
        ui.hint("  I'm not able to remember that for your account.")
    elif capability == "read_packets":
        ui.hint("  I'm not able to look things up from memory for your account.")
    elif capability == "override_conflict":
        # Silent — design says auto-skip without surfacing a hint.
        pass
    else:
        ui.hint(f"  I'm not able to do that for your account ({capability}).")


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
    user_content,  # str | list[dict] of content blocks
    model: str,
    span_name: str = "claude_call",
    stream_to_stdout: bool = False,
    max_tokens: int = 4096,
) -> str:
    """Direct Anthropic SDK call via messages.stream. Returns full text.

    `user_content` may be a plain string or a list of Anthropic content blocks
    (e.g. mixed `{type: "text"}` and `{type: "image"}` for multimodal turns).
    """
    # For tracing, summarize content blocks compactly (don't log base64 bytes).
    if isinstance(user_content, list):
        traced_user = _summarize_content_blocks_for_trace(user_content)
    else:
        traced_user = user_content

    with tracing.observation(
        span_name,
        as_type="generation",
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": traced_user},
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
            messages=[{"role": "user", "content": user_content}],
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


def _summarize_content_blocks_for_trace(blocks: list[dict]) -> str:
    """Render content blocks for the Langfuse trace without dumping base64."""
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image":
            src = b.get("source", {})
            media = src.get("media_type", "image/*")
            data = src.get("data", "")
            parts.append(f"[image:{media} bytes_b64={len(data)}]")
        else:
            parts.append(f"[block:{b.get('type', '?')}]")
    return "\n".join(parts)


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
            s_bias = strength_bias(strength, config["strength_additive_bias_scale"])
            a_bias = authoritativeness_bias(
                pkt.authoritativeness,
                config.get("authoritativeness_bias_scale", 0.001),
            )
            final = base_score + s_bias + a_bias

            considered.append({
                "id": pkt.id,
                "topics": [t.text for t in pkt.topics],
                "rrf": round(base_score, 6),
                "strength": round(strength, 3),
                "authoritativeness": round(float(pkt.authoritativeness or 0.0), 3),
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
                ui.memory_recall(
                    "{gist} final={final:.3f} rrf={rrf:.3f} strength_bias={sb:.3f} auth_bias={ab:.3f}".format(
                        gist=pkt.content.gist or first_topic,
                        final=final, rrf=base_score, sb=s_bias, ab=a_bias,
                    )
                )
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
# Write-path debug printing (gated on debug_print_write_path)
# -----------------------------------------------------------------------------

_NK_SNIPPET_CHARS = 100
_GIST_SNIPPET_CHARS = 90


def _snippet(text: str, n: int) -> str:
    s = (text or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _print_write_path_header(
    nk_index: int,
    nk_content: str,
    topic_source: str,
    topic_text: str | None,
    ingest: bool = False,
    source_url: str | None = None,
) -> None:
    tag = "[write-path · ingest]" if ingest else "[write-path]"
    print(
        f"\n{tag} new_knowledge #{nk_index}: "
        f'"{_snippet(nk_content, _NK_SNIPPET_CHARS)}"',
        flush=True,
    )
    if source_url:
        print(f"{tag} source_url: {source_url}", flush=True)
    topic_line = f'"{topic_text}"' if topic_text else "(none)"
    print(
        f"{tag} topic_vec source: {topic_source} → {topic_line}",
        flush=True,
    )


def _print_packet_candidates(
    ranked: list[tuple], threshold: float, ingest: bool = False
) -> None:
    """ranked: list of (packet, score, best_facet_text)."""
    tag = "[write-path · ingest]" if ingest else "[write-path]"
    if not ranked:
        print(f"{tag} candidates: (store is empty)", flush=True)
        return
    print(
        f"{tag} candidates ranked by max-facet cosine (top {len(ranked)}, "
        f"existing_packet_match_threshold={threshold:.2f}):",
        flush=True,
    )
    for pkt, score, facet in ranked:
        mark = "→" if score >= threshold else " "
        gist = _snippet(pkt.content.gist or "(no gist)", _GIST_SNIPPET_CHARS)
        print(
            f"  {mark} {pkt.id} (score {score:.4f}) facet={facet!r}\n"
            f"      gist: {gist}",
            flush=True,
        )


def _print_packet_decision(
    best_pkt, best_score: float, threshold: float, route: str, reason: str,
    ingest: bool = False,
) -> None:
    tag = "[write-path · ingest]" if ingest else "[write-path]"
    pkt_str = best_pkt.id if best_pkt is not None else "none"
    print(
        f"{tag} decision: {pkt_str} (best score {best_score:.4f} "
        f"{'≥' if best_score >= threshold else '<'} {threshold:.4f}) "
        f"→ route: {route}\n"
        f"{tag} reason: {reason}",
        flush=True,
    )


def _print_scratchpad_candidates(
    ranked: list[tuple], threshold: float
) -> None:
    """ranked: list of (entry, score)."""
    if not ranked:
        print(
            f"[write-path] scratchpad candidates: (empty; "
            f"scratchpad_promote_threshold={threshold:.2f})",
            flush=True,
        )
        return
    print(
        f"[write-path] scratchpad candidates ranked by topic cosine "
        f"(top {len(ranked)}, scratchpad_promote_threshold={threshold:.2f}):",
        flush=True,
    )
    for entry, score in ranked:
        mark = "→" if score >= threshold else " "
        print(
            f"  {mark} {entry.id} (score {score:.4f}) "
            f'topic="{_snippet(entry.topic, _GIST_SNIPPET_CHARS)}"',
            flush=True,
        )


def _print_url_route_check(source_url: str, matched_pkt) -> None:
    print(
        f"[write-path · ingest] URL-route check for source_url={source_url}",
        flush=True,
    )
    if matched_pkt is not None:
        print(
            f"  → matched {matched_pkt.id} (already has this URL in source_urls); "
            f"route: integrate (URL-deterministic, overrides cosine)",
            flush=True,
        )
    else:
        print(
            f"  → no packet has this URL in source_urls; falling through to cosine route",
            flush=True,
        )


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
) -> dict:
    """Single retrieval and topic-classification path.

    event_kind is "partial" or "submit"; determines trace name.
    Reads existing session topics + entities, calls extract_partial, guards
    on (is_meaningful AND (has_new_topic OR has_new_entities)), retrieves.

    Returns the extraction result so the caller (run_session) can plumb
    ingest_intent into chat_turn on submit events.
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
        ) or bool(result.get("is_ingest_request"))

        tracing.update(root, output={
            "is_meaningful": result["is_meaningful"],
            "has_new_topic": result["has_new_topic"],
            "has_new_entities": result["has_new_entities"],
            "topic_summary": result["topic_summary"],
            "entities": result["entities"],
            "is_ingest_request": result.get("is_ingest_request", False),
            "urls": result.get("urls", []),
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
            return result

        if result["has_new_topic"] and result["topic_summary"] and result["topic_vec"] is not None:
            session.add_topic(result["topic_summary"], result["topic_vec"])

        newly_loaded: list[dict] = []
        if result["topic_vec"] is not None and result["topic_summary"]:
            if not session.user.can("read_packets"):
                _capability_denied(session.user, "read_packets")
            else:
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
            if result.get("is_ingest_request"):
                bits.append(f"ingest request urls={result.get('urls')}")
            change_str = "; ".join(bits) if bits else "(extracted, no novelty flags?)"
            print(f"{prefix} {word_count} words → {change_str}", flush=True)
            for nl in newly_loaded:
                print(
                    f"  loaded {nl['id']} (score {nl['score']:.4f}) [{nl['slice']}] \"{nl['topic']}\"",
                    flush=True,
                )

        return result


# -----------------------------------------------------------------------------
# Write path helpers
# -----------------------------------------------------------------------------

async def _create_packet_from_seed(
    seed_topic: str,
    seed_content: str,
    store: PacketStore,
    config: dict,
    session_id: str,
    writer: Identity,
    source_url: str | None = None,
    initial_images: list[PacketImage] | None = None,
) -> Packet | None:
    if not writer.can("create_packet"):
        _capability_denied(writer, "create_packet")
        return None

    # Build the source record up front so we can pass its trust into the prompt.
    if source_url:
        d_auth = resolve_domain_authoritativeness(source_url, config)
        initial_source = PacketSource.from_url(source_url, d_auth, writer)
    else:
        initial_source = PacketSource.from_conversation(writer)
    seed_eff = float(initial_source.effective_authoritativeness)

    prompt = render_new_packet_prompt(
        seed_topic=seed_topic,
        seed_content=seed_content,
        seed_source_url=source_url,
        seed_authoritativeness=seed_eff,
        writer_role=writer.role,
        image_manifest_items=initial_images,
    )
    raw = await _claude_call(
        system_prompt="You are an expert at extracting clean knowledge from raw conversation excerpts.",
        user_content=prompt,
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
        sources=[initial_source],
        images=initial_images or None,
    )
    pkt.source_session_ids.append(session_id)
    pkt.record_event("write", config["strength_weights"]["write"])
    store.add(pkt)
    ui.memory_saved(pkt.content.gist or (pkt.topics[0].text if pkt.topics else ""))
    tracing.score(
        name="write", value=1,
        comment=f"created {pkt.id} trust={pkt.authoritativeness:.2f}",
    )
    if source_url:
        tracing.score(
            name="ingest", value=1,
            comment=f"{source_url} → {pkt.id} trust={pkt.authoritativeness:.2f}",
        )
    return pkt


async def integrate_into_packet(
    packet: Packet,
    new_content: str,
    store: PacketStore,
    config: dict,
    writer: Identity,
    source_url: str | None = None,
    new_images: list[PacketImage] | None = None,
) -> bool:
    if not writer.can("integrate_packet"):
        _capability_denied(writer, "integrate_packet")
        return False

    # Resolve trust for this incoming write.
    d_auth = resolve_domain_authoritativeness(source_url, config) if source_url else None
    new_eff = effective_authoritativeness(writer.role_authoritativeness, d_auth)
    existing_auth = float(getattr(packet, "authoritativeness", 0.0) or 0.0)

    facets_str = ", ".join(f'"{t.text}"' for t in packet.topics)
    entities_str = ", ".join(packet.entities[:30])
    # The integrate LLM needs to see the full image manifest (existing + new)
    # so it can preserve coco-img:<id> refs correctly. Render but don't commit
    # `new_images` to packet.images until after a successful integration.
    combined_image_manifest = list(getattr(packet, "images", []) or []) + list(new_images or [])
    existing_urls = packet.source_urls() if hasattr(packet, "source_urls") else None
    prompt = render_integrate_prompt(
        existing_facets=facets_str,
        existing_entities=entities_str,
        existing_content=packet.content.full or packet.content.summary or packet.content.gist,
        existing_source_urls=existing_urls,
        existing_authoritativeness=existing_auth,
        new_content=new_content,
        new_source_url=source_url,
        new_authoritativeness=new_eff,
        writer_role=writer.role,
        image_manifest_items=combined_image_manifest,
    )
    raw = await _claude_call(
        system_prompt="You are an expert at editing and integrating knowledge concisely.",
        user_content=prompt,
        model=config["anthropic_model"],
        span_name="integrate_on_write",
    )
    try:
        integrated = parse_integration_response(raw)
    except Exception as e:
        print(f"\n[integrate parse error: {e}]")
        return False

    trust_resolution = integrated.get("trust_resolution", "new_wins")
    tracing.score(
        name="trust_resolution", value=1,
        comment=f"{trust_resolution} new={new_eff:.2f} existing={existing_auth:.2f}",
    )

    # Equal-trust contradictions are the only branch that escalates to the user;
    # gated by override_conflict so a viewer can never overwrite an admin packet
    # via the y/N prompt.
    if trust_resolution == "equal_escalate" and integrated.get("conflict_detected"):
        if not writer.can("override_conflict"):
            tracing.score(
                name="auto_skipped_conflict", value=1,
                comment=f"role={writer.role}",
            )
            return False
        desc = integrated.get("conflicting_excerpts") or "(no description)"
        topic_name = packet.topics[0].text if packet.topics else packet.id
        print(f"\n[Conflict in packet '{topic_name}']")
        print(f"  Description: {desc}")
        print(f"  Existing trust: {existing_auth:.2f}  |  New trust: {new_eff:.2f}")
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

    # Append the PacketSource for this write. The aggregate authoritativeness
    # rises if new_eff is higher (monotone-up). Per-write provenance is preserved
    # in the source list even when an aggregate doesn't move.
    if source_url:
        packet.add_source(PacketSource.from_url(source_url, d_auth, writer))
    else:
        packet.add_source(PacketSource.from_conversation(writer))

    # Commit new images now that integration succeeded. Append unconditionally —
    # the integrate LLM may have dropped some refs in content.full; those become
    # orphan images (tolerated; future GC pass cleans them up).
    if new_images:
        for img in new_images:
            packet.add_image(img)

    packet.record_event("write", config["strength_weights"]["write"])
    store.save(packet)
    ui.memory_updated(packet.content.gist or (packet.topics[0].text if packet.topics else ""))
    tracing.score(
        name="write", value=1,
        comment=f"updated {packet.id} trust={packet.authoritativeness:.2f}",
    )
    if source_url:
        tracing.score(
            name="ingest", value=1,
            comment=f"{source_url} → {packet.id} trust={packet.authoritativeness:.2f}",
        )
    return True


# -----------------------------------------------------------------------------
# URL ingestion helpers
# -----------------------------------------------------------------------------

def _materialize_packet_images(
    content: str,
    fetch_images: dict[str, fetch_module.ImageBlob],
    source_url: str | None,
) -> tuple[str, list[PacketImage]]:
    """For each [IMG_n] in `content` that maps to a fetch image, mint a PacketImage
    and rewrite the token to `![alt](coco-img:img_<new_id>)`. Stray tokens are stripped.

    Returns (rewritten_content, list_of_minted_PacketImages).
    """
    minted: list[PacketImage] = []
    stray: list[str] = []

    def repl(m: re.Match) -> str:
        key = f"IMG_{m.group(1)}"
        blob = fetch_images.get(key)
        if blob is None:
            stray.append(key)
            return ""
        img = PacketImage.new(
            alt=blob.alt,
            mime=blob.mime,
            data_b64=blob.data_b64,
            dimensions=blob.dimensions,
            source_url=source_url or blob.src,
        )
        minted.append(img)
        alt_safe = (blob.alt or img.id).replace("]", "").replace("[", "")
        return f"![{alt_safe}](coco-img:{img.id})"

    new_content = _IMG_PLACEHOLDER_RE.sub(repl, content)
    if stray:
        try:
            print(f"[ingest warning] stray image placeholders dropped: {stray}", flush=True)
        except Exception:
            pass
    return new_content, minted


async def _handle_ingest(
    user_message: str,
    urls: list[str],
    session: Session,
    store: PacketStore,
    scratchpad: Scratchpad,
    config: dict,
    session_n: int,
) -> str:
    """Fetch URLs, run main reply with the fetched content, write to memory."""
    with tracing.observation(
        "url_ingest",
        as_type="span",
        input={"urls": urls, "user_message": user_message},
    ) as span:
        # Fetch all URLs concurrently
        fetch_tasks = [fetch_module.fetch_url(u, config) for u in urls]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        successes: list[fetch_module.FetchResult] = []
        failures: list[tuple[str, str]] = []
        for url, r in zip(urls, results):
            if isinstance(r, fetch_module.FetchError):
                failures.append((url, str(r)))
            elif isinstance(r, Exception):
                failures.append((url, f"unexpected: {r}"))
            else:
                successes.append(r)

        if not successes:
            tracing.update(span, output={"failures": failures, "successes": 0})
            ui.coco_label(admin_mode=session.admin_mode)
            reasons = "; ".join(f"{u}: {r}" for u, r in failures)
            msg = f"I couldn't read the link you shared. {reasons}\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            session.add_turn("user", user_message)
            session.add_turn("coco", msg.strip())
            return msg.strip()

        # Build the ingest-mode prompt with multimodal user content blocks
        session.add_turn("user", user_message)
        system_prompt = build_system_prompt(
            session.loaded_packets_list(),
            user_name=session.user.prompt_display_name(),
            ingest_mode=True,
        )
        history = session.turns[:-1][-(config["recency_window"] * 2):]
        user_blocks = build_user_content_blocks(
            history=history,
            current_message=user_message,
            loaded_packets=session.loaded_packets_list(),
            fetch_results=successes,
            failed_urls=failures or None,
            image_blocks_max=config.get("image_blocks_max_per_turn"),
        )

        ui.coco_label(admin_mode=session.admin_mode)
        raw = await _claude_call(
            system_prompt,
            user_blocks,
            config["anthropic_model"],
            span_name="main_reply",
            stream_to_stdout=True,
        )

        parsed = parse_coco_response(raw)
        reply = parsed["reply"]
        packets_used = parsed["packets_used"]
        new_knowledge = parsed["new_knowledge"]

        # Record use events on any cited packets
        for pid in packets_used:
            item = session.loaded_packets.get(pid)
            if item is not None:
                pkt = item["packet"]
                pkt.record_event("use", config["strength_weights"]["use"])
                store.save(pkt)
                tracing.score(name="use", value=1, comment=f"used {pid}")

        # Per-source image maps so each new_knowledge item materializes only the
        # images from its attributed URL (avoids cross-source leakage).
        per_source_images: dict[str, dict[str, fetch_module.ImageBlob]] = {
            fr.url: fr.images for fr in successes
        }
        merged_all_images: dict[str, fetch_module.ImageBlob] = {}
        for fr in successes:
            for k, v in fr.images.items():
                merged_all_images.setdefault(k, v)

        first_success_url = successes[0].url
        url_set = set(per_source_images.keys())
        written_pkt_ids: list[str] = []
        total_minted_images = 0
        debug_wp = config.get("debug_print_write_path", False)
        threshold = config["existing_packet_match_threshold"]

        for nk_idx, nk in enumerate(new_knowledge, start=1):
            content = (nk.get("content") or "").strip()
            if not content:
                continue
            src_url = nk.get("source_url") or ""
            if src_url not in url_set:
                src_url = first_success_url
            implied_topic = (nk.get("implied_topic") or "").strip() or "ingested page"
            conflicts_with = nk.get("conflicts_with")

            # Use the source-specific image map if attribution is reliable;
            # otherwise fall back to the merged map so the LLM's choice isn't lost.
            fetch_images = per_source_images.get(src_url) or merged_all_images
            content, minted_images = _materialize_packet_images(
                content, fetch_images, source_url=src_url
            )
            total_minted_images += len(minted_images)

            if debug_wp:
                _print_write_path_header(
                    nk_idx, content,
                    topic_source="LLM-supplied implied_topic",
                    topic_text=implied_topic,
                    ingest=True,
                    source_url=src_url,
                )

            # 1. URL-deterministic route
            existing = store.find_by_source_url(src_url)
            if debug_wp:
                _print_url_route_check(src_url, existing)
            if existing is not None:
                if debug_wp:
                    _print_packet_decision(
                        existing, 1.0, threshold,
                        route="integrate (URL-deterministic)",
                        reason=(
                            f"packet.source_urls already contains {src_url!r}; "
                            "re-ingest routes to the same packet regardless of cosine"
                        ),
                        ingest=True,
                    )
                ok = await integrate_into_packet(
                    existing, content, store, config, writer=session.user,
                    source_url=src_url, new_images=minted_images,
                )
                if ok:
                    written_pkt_ids.append(existing.id)
                    if existing.id not in session.loaded_packets:
                        session.loaded_packets[existing.id] = {"packet": existing, "slice": "full"}
                continue

            # 2. Conflict-named route
            if conflicts_with and conflicts_with in session.loaded_packets:
                target = session.loaded_packets[conflicts_with]["packet"]
                if debug_wp:
                    _print_packet_decision(
                        target, 1.0, threshold,
                        route="integrate (conflict-named)",
                        reason=(
                            f"LLM declared conflicts_with={target.id} in metadata; "
                            "merge directly into the named packet"
                        ),
                        ingest=True,
                    )
                ok = await integrate_into_packet(
                    target, content, store, config, writer=session.user,
                    source_url=src_url, new_images=minted_images,
                )
                if ok:
                    written_pkt_ids.append(target.id)
                continue

            # 3. Standard write-path (no scratchpad-only for ingest)
            topic_vec_for_match = embed(implied_topic, config["embedding_model"])
            all_packets = store.all()
            if debug_wp:
                ranked = rank_packet_facet_candidates(topic_vec_for_match, all_packets, top_n=5)
                _print_packet_candidates(ranked, threshold, ingest=True)
            best_pkt, best_score = best_packet_facet_match(topic_vec_for_match, all_packets)

            if best_pkt is not None and best_score >= threshold:
                if debug_wp:
                    _print_packet_decision(
                        best_pkt, best_score, threshold,
                        route="integrate (cosine)",
                        reason=(
                            f"max-facet cosine {best_score:.4f} ≥ threshold "
                            f"{threshold:.4f}; merge ingested content into this packet"
                        ),
                        ingest=True,
                    )
                ok = await integrate_into_packet(
                    best_pkt, content, store, config, writer=session.user,
                    source_url=src_url, new_images=minted_images,
                )
                if ok:
                    written_pkt_ids.append(best_pkt.id)
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
            else:
                if debug_wp:
                    _print_packet_decision(
                        best_pkt, best_score if best_pkt else -1.0, threshold,
                        route="new packet",
                        reason=(
                            f"no candidate cleared the threshold "
                            f"({best_score:.4f} < {threshold:.4f}); ingest skips "
                            "scratchpad and creates a fresh packet directly "
                            "(fetched pages are higher-weight evidence than passing conversation)"
                        ),
                        ingest=True,
                    )
                new_pkt = await _create_packet_from_seed(
                    seed_topic=implied_topic,
                    seed_content=content,
                    store=store,
                    config=config,
                    session_id=session.id,
                    writer=session.user,
                    source_url=src_url,
                    initial_images=minted_images,
                )
                if new_pkt is not None:
                    written_pkt_ids.append(new_pkt.id)
                    session.loaded_packets[new_pkt.id] = {"packet": new_pkt, "slice": "full"}

        tracing.update(span, output={
            "successes": [fr.url for fr in successes],
            "failures": failures,
            "written_packets": written_pkt_ids,
            "materialized_image_count": total_minted_images,
        })

        session.add_turn("coco", reply)
        return reply


# -----------------------------------------------------------------------------
# Document upload
# -----------------------------------------------------------------------------

async def _handle_document_upload(
    user_message: str,
    file_paths: list[str],
    session: Session,
    store: PacketStore,
    scratchpad: Scratchpad,
    config: dict,
    session_n: int,
) -> str:
    """Read local files (PDF / DOCX / PPTX / text / markdown) as a stream of
    DocumentChunks, batch them through the main-reply LLM, and route each
    new_knowledge item through the standard write path.
    """
    with tracing.observation(
        "document_upload",
        as_type="span",
        input={"file_paths": file_paths, "user_message": user_message},
    ) as span:
        session.add_turn("user", user_message)
        batch_size = int(config.get("ingest_doc_batch_chunks", 10))
        threshold = config["existing_packet_match_threshold"]
        debug_wp = config.get("debug_print_write_path", False)

        # Per-file accumulators (surfaced in the final summary and in tracing).
        total_pages = 0
        total_chunks = 0
        total_batches = 0
        total_new_packets = 0
        total_updated_packets = 0
        total_skipped_chunks = 0
        opened_files: list[str] = []
        open_errors: list[tuple[str, str]] = []
        collected_reply_parts: list[str] = []

        for raw_path in file_paths:
            path = documents_module.expand_path(raw_path)
            try:
                metadata = documents_module.open_metadata(path, config)
            except documents_module.DocumentReadError as e:
                open_errors.append((raw_path, str(e)))
                ui.coco_label(admin_mode=session.admin_mode)
                msg = f"I couldn't open {raw_path}: {e}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                collected_reply_parts.append(msg.strip())
                continue

            opened_files.append(metadata.filename)
            ui.coco_label(admin_mode=session.admin_mode)
            opening_line = (
                f"Reading {metadata.filename} "
                f"({metadata.format}, {metadata.size_bytes // 1024}KB, "
                f"trust {metadata.file_authoritativeness:.2f})...\n"
            )
            sys.stdout.write(opening_line)
            sys.stdout.flush()
            collected_reply_parts.append(opening_line.strip())

            file_new_pkts = 0
            file_updated_pkts = 0
            file_skipped = 0
            file_chunks = 0
            batch: list = []
            pages_seen: set[int] = set()

            async def _flush_batch(batch_chunks: list, batch_no: int) -> None:
                nonlocal file_new_pkts, file_updated_pkts, file_skipped, total_batches
                if not batch_chunks:
                    return
                total_batches += 1

                # Framing.
                meta_for_prompt = {
                    "filename": metadata.filename,
                    "format": metadata.format,
                    "document_type": metadata.document_type,
                    "file_authoritativeness": metadata.file_authoritativeness,
                }
                system_prompt = build_system_prompt(
                    session.loaded_packets_list(),
                    user_name=session.user.prompt_display_name(),
                    upload_mode=True,
                )
                user_blocks = build_document_batch_user_blocks(
                    metadata_dict=meta_for_prompt,
                    chunks=batch_chunks,
                    batch_index=batch_no,
                    total_pages_seen=len(pages_seen),
                )

                raw = await _claude_call(
                    system_prompt,
                    user_blocks,
                    config["anthropic_model"],
                    span_name="document_batch_reply",
                    stream_to_stdout=True,
                )
                parsed = parse_coco_response(raw)
                collected_reply_parts.append(parsed["reply"])

                # Route each new_knowledge item.
                chunk_by_ref = {c.chunk_ref(): c for c in batch_chunks}
                touched_this_batch = set(chunk_by_ref.keys())

                for nk in parsed["new_knowledge"]:
                    content = (nk.get("content") or "").strip()
                    if not content:
                        continue
                    chunk_ref = (nk.get("chunk_ref") or "").strip()
                    src_chunk = chunk_by_ref.get(chunk_ref)
                    if src_chunk is None:
                        # Soft match on page number.
                        if chunk_ref.startswith("P"):
                            try:
                                page_num = int(chunk_ref[1:].split(".")[0])
                            except ValueError:
                                page_num = None
                            if page_num is not None:
                                for c in batch_chunks:
                                    if c.page_number == page_num:
                                        src_chunk = c
                                        break
                        if src_chunk is None:
                            src_chunk = batch_chunks[0]
                            tracing.score(
                                name="document_chunk_ref_missing",
                                value=1,
                                comment=f"chunk_ref={chunk_ref!r}",
                            )

                    implied_topic = (nk.get("implied_topic") or "").strip() or (
                        f"content from {metadata.filename}"
                    )

                    # Track which chunks the LLM chose to write.
                    touched_this_batch.discard(src_chunk.chunk_ref())

                    # Build the document source for THIS write.
                    doc_source = PacketSource.from_document(
                        filename=metadata.filename,
                        document_type=metadata.document_type,
                        page_number=src_chunk.page_number,
                        paragraph_index=src_chunk.paragraph_index,
                        file_authoritativeness=metadata.file_authoritativeness,
                        writer=session.user,
                    )

                    if debug_wp:
                        _print_write_path_header(
                            0, content,
                            topic_source="LLM-supplied implied_topic",
                            topic_text=implied_topic,
                            ingest=True,
                            source_url=f"{metadata.filename}#{src_chunk.chunk_ref()}",
                        )

                    topic_vec_for_match = embed(implied_topic, config["embedding_model"])
                    all_packets = store.all()
                    if debug_wp:
                        ranked = rank_packet_facet_candidates(
                            topic_vec_for_match, all_packets, top_n=5
                        )
                        _print_packet_candidates(ranked, threshold, ingest=True)
                    best_pkt, best_score = best_packet_facet_match(
                        topic_vec_for_match, all_packets
                    )

                    if best_pkt is not None and best_score >= threshold:
                        if debug_wp:
                            _print_packet_decision(
                                best_pkt, best_score, threshold,
                                route="integrate (cosine)",
                                reason=(
                                    f"max-facet cosine {best_score:.4f} ≥ threshold; "
                                    f"merge chunk from {metadata.filename}"
                                    f"#{src_chunk.chunk_ref()}"
                                ),
                                ingest=True,
                            )
                        ok = await _integrate_with_document_source(
                            best_pkt, content, doc_source, session.user,
                            store, config,
                        )
                        if ok:
                            file_updated_pkts += 1
                            if best_pkt.id not in session.loaded_packets:
                                strength = compute_strength(
                                    best_pkt.strength_events,
                                    config["strength_weights"],
                                    config["strength_half_life_days"],
                                )
                                slice_type = slice_for_strength(
                                    strength,
                                    config["band_gist_max"],
                                    config["band_summary_max"],
                                )
                                session.loaded_packets[best_pkt.id] = {
                                    "packet": best_pkt, "slice": slice_type,
                                }
                        continue

                    # New packet.
                    if debug_wp:
                        _print_packet_decision(
                            best_pkt, best_score if best_pkt else -1.0, threshold,
                            route="new packet",
                            reason=(
                                f"no candidate cleared the threshold "
                                f"({best_score:.4f} < {threshold:.4f}); "
                                "upload creates a fresh packet"
                            ),
                            ingest=True,
                        )
                    new_pkt = await _create_packet_from_document(
                        seed_topic=implied_topic,
                        seed_content=content,
                        seed_source=doc_source,
                        writer=session.user,
                        store=store,
                        config=config,
                        session_id=session.id,
                    )
                    if new_pkt is not None:
                        file_new_pkts += 1
                        session.loaded_packets[new_pkt.id] = {
                            "packet": new_pkt, "slice": "full",
                        }

                # Untouched chunks in this batch → count as filler-skipped.
                file_skipped += len(touched_this_batch)

            # Stream chunks from the reader; batch and flush.
            try:
                async for chunk in documents_module.read_document(metadata, config):
                    file_chunks += 1
                    pages_seen.add(chunk.page_number)
                    batch.append(chunk)
                    if len(batch) >= batch_size:
                        await _flush_batch(batch, total_batches + 1)
                        batch = []
                if batch:
                    await _flush_batch(batch, total_batches + 1)
                    batch = []
            except documents_module.DocumentReadError as e:
                err = f"[document read error: {e}]"
                sys.stdout.write(err + "\n")
                sys.stdout.flush()
                collected_reply_parts.append(err)

            total_pages += len(pages_seen)
            total_chunks += file_chunks
            total_new_packets += file_new_pkts
            total_updated_packets += file_updated_pkts
            total_skipped_chunks += file_skipped

            per_file_line = (
                f"\n{metadata.filename}: {len(pages_seen)} pages, "
                f"{file_chunks} chunks → "
                f"{file_new_pkts} new packet(s), {file_updated_pkts} updated, "
                f"{file_skipped} chunks skipped as filler.\n"
            )
            sys.stdout.write(per_file_line)
            sys.stdout.flush()
            collected_reply_parts.append(per_file_line.strip())

        summary_reply = (
            f"Read {len(opened_files)} file(s): {', '.join(opened_files) or '(none)'}. "
            f"{total_pages} pages, {total_chunks} chunks → "
            f"{total_new_packets} new packet(s), {total_updated_packets} updated, "
            f"{total_skipped_chunks} skipped."
        )
        if open_errors:
            summary_reply += (
                " Errors: "
                + "; ".join(f"{p}: {e}" for p, e in open_errors)
            )

        final_reply = "\n".join(collected_reply_parts + [summary_reply])
        session.add_turn("coco", final_reply)

        tracing.update(span, output={
            "opened_files": opened_files,
            "open_errors": open_errors,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_batches": total_batches,
            "new_packets": total_new_packets,
            "updated_packets": total_updated_packets,
            "skipped_chunks": total_skipped_chunks,
        })
        return final_reply


async def _integrate_with_document_source(
    packet: Packet,
    new_content: str,
    doc_source: PacketSource,
    writer: Identity,
    store: PacketStore,
    config: dict,
) -> bool:
    """Merge document-sourced content into an existing packet.

    Structurally identical to integrate_into_packet, but the PacketSource
    appended on commit is a document-type source (not URL / conversation).
    Trust and conflict resolution work the same way; the `source_trust` term
    is the document's file_authoritativeness.
    """
    if not writer.can("integrate_packet"):
        _capability_denied(writer, "integrate_packet")
        return False

    new_eff = float(doc_source.effective_authoritativeness or 0.0)
    existing_auth = float(getattr(packet, "authoritativeness", 0.0) or 0.0)

    facets_str = ", ".join(f'"{t.text}"' for t in packet.topics)
    entities_str = ", ".join(packet.entities[:30])
    combined_image_manifest = list(getattr(packet, "images", []) or [])
    existing_urls = packet.source_urls() if hasattr(packet, "source_urls") else None

    # Fabricate a display-only source label for the integrate prompt so the
    # LLM can reference provenance in the merged text ("per handbook.pdf p14").
    src_label = f"{doc_source.filename}#p{doc_source.page_number}"
    if doc_source.paragraph_index is not None:
        src_label += f".{doc_source.paragraph_index}"

    prompt = render_integrate_prompt(
        existing_facets=facets_str,
        existing_entities=entities_str,
        existing_content=packet.content.full or packet.content.summary or packet.content.gist,
        existing_source_urls=existing_urls,
        existing_authoritativeness=existing_auth,
        new_content=new_content,
        new_source_url=src_label,
        new_authoritativeness=new_eff,
        writer_role=writer.role,
        image_manifest_items=combined_image_manifest,
    )

    raw = await _claude_call(
        system_prompt="You are an expert at editing and integrating knowledge concisely.",
        user_content=prompt,
        model=config["anthropic_model"],
        span_name="integrate_on_write",
    )
    try:
        integrated = parse_integration_response(raw)
    except Exception as e:
        print(f"\n[integrate parse error: {e}]")
        return False

    trust_resolution = integrated.get("trust_resolution", "new_wins")
    tracing.score(
        name="trust_resolution", value=1,
        comment=f"{trust_resolution} new={new_eff:.2f} existing={existing_auth:.2f}",
    )

    if trust_resolution == "equal_escalate" and integrated.get("conflict_detected"):
        if not writer.can("override_conflict"):
            tracing.score(
                name="auto_skipped_conflict", value=1,
                comment=f"role={writer.role}",
            )
            return False
        desc = integrated.get("conflicting_excerpts") or "(no description)"
        topic_name = packet.topics[0].text if packet.topics else packet.id
        print(f"\n[Conflict in packet '{topic_name}']")
        print(f"  Description: {desc}")
        print(f"  Existing trust: {existing_auth:.2f}  |  New trust: {new_eff:.2f}")
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
    packet.add_source(doc_source)
    packet.record_event("write", config["strength_weights"]["write"])
    store.save(packet)
    ui.memory_updated(packet.content.gist or (packet.topics[0].text if packet.topics else ""))
    tracing.score(
        name="write", value=1,
        comment=f"updated {packet.id} trust={packet.authoritativeness:.2f}",
    )
    tracing.score(
        name="document_write", value=1,
        comment=f"{src_label} → {packet.id} trust={packet.authoritativeness:.2f}",
    )
    return True


async def _create_packet_from_document(
    seed_topic: str,
    seed_content: str,
    seed_source: PacketSource,
    writer: Identity,
    store: PacketStore,
    config: dict,
    session_id: str,
) -> Packet | None:
    """Create a fresh packet from a document chunk.

    Mirrors _create_packet_from_seed but attaches a document-type PacketSource
    (not URL / conversation).
    """
    if not writer.can("create_packet"):
        _capability_denied(writer, "create_packet")
        return None

    seed_eff = float(seed_source.effective_authoritativeness or 0.0)

    prompt = render_new_packet_prompt(
        seed_topic=seed_topic,
        seed_content=seed_content,
        seed_source_url=None,   # document sources render their own label below
        seed_authoritativeness=seed_eff,
        writer_role=writer.role,
        image_manifest_items=None,
    )
    raw = await _claude_call(
        system_prompt="You are an expert at extracting clean knowledge from raw conversation excerpts.",
        user_content=prompt,
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
        sources=[seed_source],
    )
    pkt.source_session_ids.append(session_id)
    pkt.record_event("write", config["strength_weights"]["write"])
    store.add(pkt)
    ui.memory_saved(pkt.content.gist or (pkt.topics[0].text if pkt.topics else ""))
    tracing.score(
        name="write", value=1,
        comment=f"created {pkt.id} trust={pkt.authoritativeness:.2f}",
    )
    tracing.score(
        name="document_write", value=1,
        comment=(
            f"{seed_source.filename}#p{seed_source.page_number}"
            + (
                f".{seed_source.paragraph_index}"
                if seed_source.paragraph_index is not None else ""
            )
            + f" → {pkt.id} trust={pkt.authoritativeness:.2f}"
        ),
    )
    return pkt


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
    ingest_intent: dict | None = None,
    upload_intent: dict | None = None,
) -> str:
    _reset_denial_hints()
    with tracing.observation(
        "chat_turn",
        as_type="span",
        input={
            "user_message": user_message,
            "ingest_intent": ingest_intent,
            "upload_intent": upload_intent,
        },
        metadata={
            "session_id": session.id,
            "session_n": session_n,
            "turn_index": len(session.turns),
            "user_role": session.user.role,
            "user_role_authoritativeness": session.user.role_authoritativeness,
            "user_provider": session.user.provider,
        },
    ) as root:
        ingest_requested = (
            config.get("ingest_enabled", True)
            and ingest_intent
            and ingest_intent.get("is_ingest_request")
            and ingest_intent.get("urls")
        )
        if ingest_requested and not session.user.can("skill.fetch_url"):
            _capability_denied(session.user, "skill.fetch_url")
            ingest_requested = False

        upload_requested = (
            config.get("ingest_doc_enabled", True)
            and upload_intent
            and upload_intent.get("is_upload_request")
            and upload_intent.get("file_paths")
        )
        if upload_requested and not session.user.can("skill.upload_document"):
            _capability_denied(session.user, "skill.upload_document")
            upload_requested = False

        if upload_requested:
            reply = await _handle_document_upload(
                user_message,
                upload_intent["file_paths"],
                session,
                store,
                scratchpad,
                config,
                session_n,
            )
        elif ingest_requested:
            reply = await _handle_ingest(
                user_message,
                ingest_intent["urls"],
                session,
                store,
                scratchpad,
                config,
                session_n,
            )
        else:
            reply = await _chat_turn_inner(
                user_message, session, store, scratchpad, config, session_n
            )
        tracing.update(root, output={
            "reply": reply,
            "loaded_packets": list(session.loaded_packets.keys()),
            "ingest": bool(ingest_intent and ingest_intent.get("is_ingest_request")),
            "upload": bool(upload_intent and upload_intent.get("is_upload_request")),
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
        session.loaded_packets_list(),
        user_name=session.user.prompt_display_name(),
    )
    history = session.turns[:-1][-(config["recency_window"] * 2):]
    # Use the content-blocks builder so full-slice loaded packets contribute
    # their PacketImage bytes as multimodal blocks. With no images, this
    # collapses to a single text block (functionally identical to the prior
    # string-based path).
    user_content = build_user_content_blocks(
        history=history,
        current_message=user_message,
        loaded_packets=session.loaded_packets_list(),
        fetch_results=None,
        failed_urls=None,
        image_blocks_max=config.get("image_blocks_max_per_turn"),
    )
    # Optimization: if no image blocks are needed, send the plain string
    # (keeps the SDK / trace shape simple for the common case).
    if len(user_content) == 1 and user_content[0].get("type") == "text":
        user_content = user_content[0]["text"]

    ui.coco_label(admin_mode=session.admin_mode)
    raw = await _claude_call(
        system_prompt,
        user_content,
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

    debug_wp = config.get("debug_print_write_path", False)
    threshold = config["existing_packet_match_threshold"]
    sp_threshold = config["scratchpad_promote_threshold"]

    for nk_idx, nk in enumerate(new_knowledge, start=1):
        content = (nk.get("content") or "").strip()
        if not content:
            continue
        conflicts_with = nk.get("conflicts_with")

        if conflicts_with and conflicts_with in session.loaded_packets:
            target = session.loaded_packets[conflicts_with]["packet"]
            if debug_wp:
                _print_write_path_header(
                    nk_idx, content,
                    topic_source="LLM-flagged conflicts_with",
                    topic_text=(target.topics[0].text if target.topics else target.id),
                )
                _print_packet_decision(
                    target, 1.0, threshold,
                    route="integrate (conflict-named)",
                    reason=f"LLM declared this knowledge conflicts with loaded packet {target.id}",
                )
            await integrate_into_packet(
                target, content, store, config, writer=session.user,
            )
            continue

        if topic_vec is None:
            if debug_wp:
                print(
                    f"\n[write-path] new_knowledge #{nk_idx}: skipped — "
                    "no current session topic to score against",
                    flush=True,
                )
            continue  # no current topic; can't do facet matching

        all_packets = store.all()
        if debug_wp:
            _print_write_path_header(
                nk_idx, content,
                topic_source="session.current_topic",
                topic_text=(current_topic.get("topic_text") if current_topic else None),
            )
            ranked = rank_packet_facet_candidates(topic_vec, all_packets, top_n=5)
            _print_packet_candidates(ranked, threshold)

        best_pkt, best_score = best_packet_facet_match(topic_vec, all_packets)
        if best_pkt is not None and best_score >= threshold:
            if debug_wp:
                _print_packet_decision(
                    best_pkt, best_score, threshold,
                    route="integrate",
                    reason=(
                        f"max-facet cosine {best_score:.4f} ≥ threshold "
                        f"{threshold:.4f}; merge new knowledge into this packet"
                    ),
                )
            await integrate_into_packet(
                best_pkt, content, store, config, writer=session.user,
            )
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
        if debug_wp:
            _print_packet_decision(
                best_pkt, best_score if best_pkt else -1.0, threshold,
                route="scratchpad",
                reason=(
                    f"no packet cleared cosine threshold ({best_score:.4f} < {threshold:.4f}); "
                    "checking scratchpad for a near-duplicate seed"
                ),
            )
            sp_ranked = rank_scratchpad_candidates(topic_vec, entries, top_n=3)
            _print_scratchpad_candidates(sp_ranked, sp_threshold)

        match, _ = best_scratchpad_match(topic_vec, entries, sp_threshold)
        if match is not None and session.user.can("promote_scratchpad"):
            if debug_wp:
                print(
                    f"[write-path] scratchpad decision: promote {match.id} → new packet "
                    f"(topic={match.topic!r}; raw_excerpts={len(match.raw_excerpts)})",
                    flush=True,
                )
            seed_content = "\n\n".join(match.raw_excerpts + [user_message, content])
            new_pkt = await _create_packet_from_seed(
                seed_topic=match.topic,
                seed_content=seed_content,
                store=store,
                config=config,
                session_id=session.id,
                writer=session.user,
            )
            if new_pkt is not None:
                scratchpad.remove(match.id)
                session.loaded_packets[new_pkt.id] = {"packet": new_pkt, "slice": "full"}
                if config.get("debug_print_streaming", False):
                    print(f"[promoted scratchpad entry to packet: {[t.text for t in new_pkt.topics]}]")
        else:
            if match is not None and not session.user.can("promote_scratchpad"):
                _capability_denied(session.user, "promote_scratchpad")
            if not session.user.can("write_scratchpad"):
                _capability_denied(session.user, "write_scratchpad")
                continue
            if debug_wp:
                print(
                    f"[write-path] scratchpad decision: no entry above threshold; "
                    "inserting a fresh scratchpad entry",
                    flush=True,
                )
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

async def run_session(cli_flags=None):
    config = load_config()

    # Acquire identity BEFORE opening the input stream. acquire_identity may
    # itself read from stdin (the provider prompt) and we don't want the
    # streaming reader to compete for the tty. `cli_flags` (parsed by
    # __main__) is threaded through so the local-admin short-circuit (--admin)
    # can fire before any IdP or interactive prompt runs.
    identity = await acquire_identity(config, cli_flags=cli_flags)

    store = PacketStore(config["data_dir"])
    scratchpad = Scratchpad(config["data_dir"])
    counter = SessionCounter(config["data_dir"])
    session_n = counter.increment()
    scratchpad.prune_old(session_n, config["scratchpad_discard_after_sessions"])

    # Pass config so `tracing.enabled=false` can short-circuit before we
    # import langfuse or read LANGFUSE_* env vars. Config wins over env.
    tracing_on = tracing.init(config)

    # Pre-load embedding model so the first extraction is not blocked on it.
    get_model(config["embedding_model"])

    session = Session(user=identity)

    # Unmissable warning for local admin mode — before the welcome banner so
    # the user cannot miss it, and again in the goodbye line at session end.
    if session.admin_mode:
        ui.banner_admin_warning()

    ui.banner_welcome(identity.name)

    debug_state = config.get("debug_print_state", False)
    debug_streaming = config.get("debug_print_streaming", False)
    debug_write_path = config.get("debug_print_write_path", False)
    if debug_state or debug_streaming or debug_write_path:
        ui.hint(
            f"developer mode · session {session.id} · "
            f"identity={identity.name} role={identity.role} "
            f"(auth={identity.role_authoritativeness:.2f}, provider={identity.provider}) · "
            f"{len(store.all())} packets, {len(scratchpad.all())} scratchpad · "
            f"tracing {'on' if tracing_on else 'off'}"
        )
    if debug_state:
        _print_state(session, prefix="[state · session start]")

    with tracing.session_context(
        session.id,
        user_id=(identity.email or identity.name),
        metadata=identity.trace_metadata(),
    ):
        in_flight: set[asyncio.Task] = set()

        async for event in streaming.input_stream(config, admin_mode=session.admin_mode):
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

            submit_result: dict | None = None
            try:
                submit_result = await on_text_event(
                    event.text, session, store, scratchpad, config, "submit"
                )
            except Exception as e:
                print(f"\n[submit-extraction error: {e}]")

            ingest_intent = None
            upload_intent = None
            if submit_result is not None:
                ingest_intent = {
                    "is_ingest_request": submit_result.get("is_ingest_request", False),
                    "urls": submit_result.get("urls", []),
                }
                upload_intent = {
                    "is_upload_request": submit_result.get("is_upload_request", False),
                    "file_paths": submit_result.get("file_paths", []),
                }

            try:
                await chat_turn(
                    event.text, session, store, scratchpad, config, session_n,
                    ingest_intent=ingest_intent,
                    upload_intent=upload_intent,
                )
            except Exception as e:
                print(f"\n[turn error: {e}]")

            if debug_state:
                _print_state(session, prefix="[state · after turn]")

    if debug_state or debug_streaming or debug_write_path:
        ui.hint(
            f"developer mode · session ended · "
            f"{len(session.topics)} topics, {len(session.loaded_packets)} packets loaded"
        )
    ui.banner_goodbye(admin_mode=session.admin_mode)
    tracing.flush()
