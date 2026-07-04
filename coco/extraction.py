"""Small-LM extraction of (topic_summary, entities, ingest intent) plus novelty flags.

Called by `agent.on_text_event` for every partial and submit. Decides whether
retrieval should fire based on three checks combined: substantive, has-new-topic,
has-new-entities. Also flags URL ingest requests for `chat_turn` to dispatch.
"""
import re

from .embeddings import embed
from .llm import anthropic_client
from .prompts import EXTRACTION_PROMPT, parse_extraction_response
from . import tracing


_URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+", re.IGNORECASE)


_EXTRACT_SYSTEM = "You are a precise topic and entity extractor. Output JSON only."


async def _small_lm_call(prompt: str, model: str, max_tokens: int = 1024) -> str:
    client = anthropic_client()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _format_topics(topics: list[str]) -> str:
    if not topics:
        return "(none yet)"
    return "\n".join(f"- {t}" for t in topics)


def _format_entities(entities: list[str]) -> str:
    if not entities:
        return "(none yet)"
    return ", ".join(entities)


async def extract_partial(
    partial_text: str,
    existing_topics: list[str],
    existing_entities: list[str],
    config: dict,
) -> dict:
    """Run the small LM with novelty context. Embeds topic_summary inline
    when retrieval will be triggered.

    Returns:
        {
            "is_meaningful":     bool,
            "has_new_topic":     bool,
            "has_new_entities":  bool,
            "topic_summary":     str | None,
            "entities":          list[str],
            "topic_vec":         np.ndarray | None,
            "is_ingest_request": bool,
            "urls":              list[str],
            "reason":            str | None,
        }
    """
    prompt = EXTRACTION_PROMPT.format(
        partial_text=partial_text,
        existing_topics=_format_topics(existing_topics),
        existing_entities=_format_entities(existing_entities),
    )

    with tracing.observation(
        "streaming_extraction",
        as_type="generation",
        model=config["small_lm_model"],
        input=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        metadata={
            "partial_text": partial_text,
            "existing_topics": existing_topics,
            "existing_entities": existing_entities,
        },
    ) as gen:
        raw = await _small_lm_call(prompt, config["small_lm_model"])
        tracing.update(gen, output=raw)

    regex_urls = _dedupe_urls(_URL_RE.findall(partial_text))

    try:
        d = parse_extraction_response(raw)
    except Exception as e:
        return {
            "is_meaningful": False,
            "has_new_topic": False,
            "has_new_entities": False,
            "topic_summary": None,
            "entities": [],
            "topic_vec": None,
            "is_ingest_request": False,
            "urls": regex_urls,
            "reason": f"parse_error: {e}",
        }

    is_meaningful = bool(d.get("is_meaningful", False))
    has_new_topic = bool(d.get("has_new_topic", False))
    has_new_entities = bool(d.get("has_new_entities", False))
    topic_summary = d.get("topic_summary") or None
    entities = [e.lower().strip() for e in (d.get("entities") or []) if e and e.strip()]
    reason = d.get("reason")
    is_ingest_request = bool(d.get("is_ingest_request", False))
    llm_urls = _dedupe_urls([u for u in (d.get("urls") or []) if isinstance(u, str)])
    # Union of LLM-extracted and regex-extracted URLs — regex is the backstop.
    urls = _dedupe_urls(llm_urls + regex_urls)

    # Ingest with no URLs is a small-LM bug — degrade to non-ingest.
    if is_ingest_request and not urls:
        is_ingest_request = False

    should_retrieve = is_meaningful and (has_new_topic or has_new_entities)
    # An ingest request should still trigger retrieval so related packets load.
    if is_ingest_request:
        should_retrieve = True

    topic_vec = None
    if should_retrieve and topic_summary:
        topic_vec = embed(topic_summary, config["embedding_model"])

    return {
        "is_meaningful": is_meaningful,
        "has_new_topic": has_new_topic,
        "has_new_entities": has_new_entities,
        "topic_summary": topic_summary,
        "entities": entities,
        "topic_vec": topic_vec,
        "is_ingest_request": is_ingest_request,
        "urls": urls,
        "reason": reason,
    }


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        s = (u or "").strip().rstrip(".,;:)\"'")
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out
