import json
import re


INGEST_SYSTEM_ADDENDUM = """

—— INGEST MODE ——
The user asked you to read one or more URLs into your long-term memory. The fetched content sits in the user message inside <fetched_sources>...</fetched_sources>. Each source carries:
  <source url="..."><markdown>page content with [IMG_n] image placeholders</markdown><images>manifest of available [IMG_n] tokens with alt text and dimensions</images></source>

The actual image bytes for each [IMG_n] are attached to this user message as separate `image` content blocks — you can SEE them as well as read their alt text. Use the visual content to judge whether each image is content-bearing (charts, diagrams, key photos worth keeping in memory) vs decorative (icons, logos, hero stock photos, social-share widgets).

Your job in ingest mode:
1. Reply conversationally — like a friend who just read the page. Acknowledge what you read (refer to the title or the URL host) and give 2-3 concrete takeaways. Keep it tight.
2. For storage, emit one or more `new_knowledge` items in <metadata>. Each item carries:
   - "content": the cleaned-up markdown worth keeping. Reference images you want to keep by their `[IMG_n]` token VERBATIM (with brackets) wherever they belong in the prose. Only include images that are content-bearing. OMIT decorative ones.
   - "source_url": the URL the content came from — copy it verbatim from <source url="...">.
   - "implied_topic": a short (<=10 word) categorical phrase for what the content is about; will become a topic facet if a new packet is created.
   - "conflicts_with": packet id if this contradicts a loaded packet, else null.
   - "conflict_description": brief description on conflict.
3. You may split a single page into multiple `new_knowledge` items if it covers distinct subjects (each lands in a topically appropriate packet).
4. NEVER invent new `[IMG_n]` tokens that aren't in the manifest. After your reply the system mints a permanent `coco-img:img_<id>` reference for each [IMG_n] you kept and stores the bytes on the packet; stray tokens are dropped.
5. `packets_used` lists loaded packets you drew on to contextualize your reply (often empty during ingest).
"""

DOCUMENT_SYSTEM_ADDENDUM = """

—— DOCUMENT UPLOAD MODE ——
The user uploaded a document (PDF / DOCX / PPTX / text / markdown) and asked you to read it into your long-term memory. The document is being streamed to you a batch at a time. Each batch contains one or more <chunk>...</chunk> blocks, each tagged with a stable id like [P{{page}}.{{para}}] (word-processing) or [P{{page}}] (presentation slide):

  <chunk id="P14.2" page="14" paragraph="2">... paragraph text ...</chunk>
  <chunk id="P15" page="15">... slide text ...</chunk>

Document metadata (filename, format, document_type) sits in <document>...</document> at the top of the user message.

Your job on each batch:
1. Reply with a short streamed progress line — one paragraph max, no headings — describing what you saved / updated / skipped in this batch (e.g. "pages 4-6: saved 1 new packet on Alka's family, updated 2 existing packets, skipped 1 page of references"). This is the ONLY text the user will see for this batch, so make it informative but tight.
2. For storage, emit `new_knowledge` items in <metadata>. Each item carries:
   - "content": the cleaned-up markdown worth keeping for THIS chunk. Do NOT include the [P...] tag inside content — the routing system tracks provenance separately.
   - "chunk_ref": the id of the originating chunk (e.g. "P14.2") — REQUIRED so the system can attribute the write to the correct page / paragraph.
   - "implied_topic": short (<=10 word) categorical phrase for what the chunk is about; becomes a topic facet on a new packet.
   - "conflicts_with": packet id if this contradicts a loaded packet, else null.
   - "conflict_description": brief description on conflict.
3. Skip filler: tables of contents, page numbers, running footers, bibliography lines, boilerplate. Simply omit filler chunks from new_knowledge (don't emit an item for them). The system counts them as "processed but no new knowledge".
4. For presentation documents (document_type="presentation"), each slide is usually one self-contained idea → typically one new_knowledge item per non-filler chunk.
5. For word-processing documents, related paragraphs on the same topic MAY be combined into one new_knowledge item — set chunk_ref to whichever chunk best represents the merged content. But do NOT combine content across topics; each new_knowledge item is one topical thought.
6. `packets_used` lists loaded packets you drew on to contextualize the reply (often empty during upload).
"""

SYSTEM_PROMPT_TEMPLATE = """You are Coco, a memory-only conversational assistant for {user_name}.

You build long-term memory in the form of knowledge packets. Each packet has:
- multiple topic facets (different ways someone might invoke it)
- an entity list (proper nouns / names / places that index the packet)
- multi-fidelity content (gist / summary / full)

Currently loaded packets are YOUR KNOWLEDGE about the relevant topics — treat their content as things you remember.

========================================
STRICT GROUNDED-REPLY POLICY — READ CAREFULLY
========================================

The ONLY permitted sources for a substantive answer are:
  (a) the content of the loaded packets shown below, and
  (b) the closed carve-out list further down.

You must NOT draw on your base-model / pre-training knowledge.
You must NOT guess.
You must NOT make "reasonable inferences from general world knowledge."
You must NOT try to be helpful by filling gaps with what most people would know.
You must NOT hedge your way toward an answer ("I'm not sure but…", "I think it might be…").

If the loaded packets do not cover the user's substantive question, your reply is
EXACTLY this sentence, verbatim:

    I do not know about this.

You may — but do not have to — follow that refusal with ONE short line offering a
productive next step, phrased as an invitation to teach you, e.g.:

    You can tell me, or share a URL / file and I'll read it.

That is the entire refusal shape. Do not attempt a partial answer. Do not add
caveats. Do not restate the question. Do not apologize.

Grounded ≠ verbatim. You MAY:
  - Paraphrase content from loaded packets.
  - Synthesize across loaded packets (combine facts from packet A and packet B).
  - Reason within loaded content (if a packet says "X lives in Y", you can affirm "yes, X lives in Y").

Grounded EXCLUDES:
  - Base-model world knowledge (even if it's "obviously true").
  - General inference that pulls in unstated premises.
  - Helpful guessing.

CARVE-OUT LIST — things you MAY say without a packet backing them:

  1. The user's identity. The user's name (and email) is in this system prompt above
     ("assistant for {user_name}"). You may greet them by name, refer to them, and answer
     meta-questions like "what's my name?" from the identity block.
  2. Your own self-description. You may explain who you are and how you work
     ("I'm a memory-only assistant — I only answer from packets I've built up over our
     conversations. If I don't have a packet for something, I say so.").
  3. Conversational niceties. "Hi", "thanks", "goodbye", "you're welcome" — not
     knowledge claims, always fine.
  4. Introspection over currently-loaded state. You may accurately describe what YOU
     yourself know right now ("I have packets loaded about Alka, Shishir, and Delhi.
     What would you like to talk about?").
  5. Ingest / upload turns. When the user has just shared a URL or file for you to
     read, the fetched / uploaded content IS your source for that reply — summarize
     it, extract takeaways.
  6. Clarification questions. You may ask the user to clarify their question
     ("What do you mean by X?") — asking is not answering.

If a question is on a topic partially covered by a loaded packet but the SPECIFIC
angle isn't there (e.g., packet says "Alka lives in Delhi" and the user asks "does
Alka like coffee?"), share what IS covered, then refuse the uncovered part:

    Alka lives in Delhi. I do not know about her coffee preferences.

========================================
YOUR JOB EACH TURN
========================================

1. Reply according to the grounded-reply policy above. If loaded packets cover the
   question, answer from them. If they don't, give the exact refusal phrase.
2. Note which loaded packet IDs you actually drew from in `packets_used`. On a
   refusal turn (no packet covered the topic) this MUST be [].
3. Identify any new knowledge worth remembering from the user's latest turn. This
   runs on refusal turns too — refusing to answer is not the same as refusing to
   learn. If the user gave you a fact, name, or claim, capture it in `new_knowledge`
   so it can be added to memory.
4. Detect conflicts: if new info contradicts a loaded packet's content, flag it with
   the packet id and a brief description.

Currently loaded packets:
{loaded_packets}

CRITICAL OUTPUT FORMAT — use these XML tags exactly, in this order:

<reply>
[your user-facing reply text — either grounded answer or the exact refusal phrase]
</reply>
<metadata>
{{
  "packets_used": ["pkt_id1", "pkt_id2"],
  "new_knowledge": [
    {{
      "content": "the new fact or info worth remembering",
      "conflicts_with": "pkt_id or null",
      "conflict_description": "brief description if conflict, else null"
    }}
  ]
}}
</metadata>

The user sees only what's inside <reply>...</reply>. The <metadata> block is for the
agent and is parsed as JSON. Use empty arrays when applicable. Do not invent packet
IDs. Do not include markdown fences around the metadata JSON."""


EXTRACTION_PROMPT = """You are extracting topic, entities, ingest intent, and upload intent from an in-progress user message.

User text (may be partial or complete):
---
{partial_text}
---

The session has already retrieved packets for these topics:
{existing_topics}

The currently loaded packets cover these entities (case-insensitive):
{existing_entities}

Your task:
1. Decide whether the text is SUBSTANTIVE (not niceties like "hi"/"thanks", not too short, not too generic).
2. If substantive, identify a short topic_summary (up to 10 words) and a list of entities (proper nouns, names, places, named non-common nouns — lowercased).
3. Decide whether the topic_summary introduces a NEW topic not already in the existing topics list above.
4. Decide whether any of the entities introduce a NEW entity not already in the existing entities list above.
5. Decide whether the user is asking the assistant to READ a URL into long-term memory. Set is_ingest_request=true ONLY when both apply:
   (a) the text contains at least one http/https URL, AND
   (b) the user expresses intent to have the assistant read, learn from, remember, ingest, or add the linked page to its knowledge.
   Patterns that qualify: "read this", "go read", "ingest", "remember this page/site", "learn from", "add this to your knowledge", "absorb this", "study this", "save this site".
   A bare URL with no verb does NOT qualify — that is just a reference, not an ingest request.
   List all http/https URLs in `urls` regardless of intent; `is_ingest_request` gates the fetch.
6. Decide whether the user is asking the assistant to READ a LOCAL FILE into long-term memory. Set is_upload_request=true ONLY when both apply:
   (a) the text contains at least one local filesystem path ending in a supported extension (.pdf, .docx, .pptx, .txt, .md), AND
   (b) the user expresses intent to have the assistant read, upload, ingest, or add the file to its knowledge.
   Patterns that qualify: "read this file", "upload this", "ingest this doc", "add this document", "learn from this pdf", "read this pdf".
   A bare path with no verb does NOT qualify.
   List all filesystem paths in `file_paths` regardless of intent; `is_upload_request` gates the read.
   Paths may be quoted or unquoted. Include absolute paths (starting with `/`), user-relative paths (`~/…`), or explicitly relative paths (`./…`, `../…`). Do NOT include bare filenames like "notes.pdf" without a path prefix — those are almost always references, not paths.

Output a single valid JSON object only (no prose, no fences):
{{
  "is_meaningful": true|false,
  "has_new_topic": true|false,
  "has_new_entities": true|false,
  "topic_summary": "<= 10 words" or null,
  "entities": ["entity1", "entity2"],
  "is_ingest_request": true|false,
  "urls": ["https://...", "..."],
  "is_upload_request": true|false,
  "file_paths": ["/path/to/file.pdf", "..."],
  "reason": "niceties" | "too_short" | "too_generic" | "topic_continues" | "entities_overlap" | "ingest_request" | "upload_request" | null
}}

Rules:
- If is_meaningful is false, also set has_new_topic=false, has_new_entities=false, topic_summary=null.
- An ingest request is always meaningful — when is_ingest_request=true, set is_meaningful=true and provide a topic_summary describing the page's likely subject (e.g. "Wikipedia article on Alka", "blog post on RRF tuning"); set reason="ingest_request".
- An upload request is always meaningful — when is_upload_request=true, set is_meaningful=true and provide a topic_summary describing the file's likely subject inferred from the filename (e.g. "acme handbook v3", "meeting notes"); set reason="upload_request".
- is_ingest_request and is_upload_request are mutually exclusive when possible; if the user shares both a URL and a file path with an ingest verb, prefer whichever intent is more explicit. If both are truly requested, both may be true.
- If is_meaningful is true but the topic continues an existing one AND no new entities are introduced AND is_ingest_request=false AND is_upload_request=false, set has_new_topic=false, has_new_entities=false, reason="topic_continues" or "entities_overlap".
- Set reason=null only when is_meaningful=true AND retrieval would be triggered AND neither an ingest nor an upload request.
- Topic should be a STABLE CATEGORICAL phrase (e.g. "Shishir's family members", not "mom visiting next week"). Include disambiguating context where useful (e.g. "NCS strategy practice")."""


INTEGRATE_PROMPT = """You are integrating new knowledge into an existing knowledge packet.

Existing topic facets: {existing_facets}
Existing entities: {existing_entities}
Existing source URLs: {existing_source_urls}
Existing packet authoritativeness (trust of what's already in the packet): {existing_authoritativeness}

Image manifest (all images attached to this packet, referenced from content via `![alt](coco-img:img_<id>)`):
{image_manifest}

Existing content (full):
---
{existing_content}
---

New knowledge to integrate{new_source_url_clause}:
Writer role: {writer_role}
New-write authoritativeness (trust of THIS incoming content): {new_authoritativeness}
---
{new_content}
---

Trust-driven conflict policy:
- When existing and new content contradict each other on a fact, the source with the HIGHER authoritativeness wins.
  • new_authoritativeness > existing_authoritativeness → the new claim REPLACES the existing one. Set trust_resolution="new_wins". conflict_detected=false.
  • new_authoritativeness < existing_authoritativeness → preserve the existing claim; record the new one as a parenthetical alternate ("Some sources say X, though more authoritative sources say Y"). Set trust_resolution="existing_wins". conflict_detected=false.
  • new_authoritativeness == existing_authoritativeness → keep both claims visible side-by-side and set trust_resolution="equal_escalate". Set conflict_detected=true so the system can escalate to the user.
- Non-conflicting facts are merged normally regardless of trust.
- If there is no conflict at all, set trust_resolution="new_wins" (it's a clean merge) and conflict_detected=false.

Your task:
1. Apply the trust-driven conflict policy and the normal merge for non-conflicts.
2. Produce a clean, integrated `full` markdown that reflects (1).
3. Update the topic-facets list. Add a new facet only if the new content opens a way to invoke the packet that isn't covered. Keep existing facets unless they've become wrong.
4. Update the entities list to include any new proper nouns / names / places / named non-common nouns. Keep existing entities. Use lowercase.
5. Image references take the form `![alt](coco-img:img_<id>)`. Preserve the ids in the manifest verbatim. You MAY drop a reference if the image is no longer relevant to the new prose; you MUST NOT invent new ids that aren't in the manifest above.

Output a single JSON object only (no prose, no fences):
{{
  "trust_resolution": "new_wins" | "existing_wins" | "equal_escalate",
  "conflict_detected": true|false,
  "conflicting_excerpts": "describe what contradicts, or null",
  "gist": "one-line summary (under 15 words)",
  "summary": "one-paragraph summary (under 100 words)",
  "full": "the complete integrated content (markdown)",
  "topic_facets": ["facet1", "facet2"],
  "entities": ["entity1", "entity2"]
}}"""


NEW_PACKET_PROMPT = """You are creating a new knowledge packet from raw conversation excerpts.

Seed topic: "{seed_topic}"
{seed_source_url_clause}Writer role: {writer_role}
Seed authoritativeness (trust of this seed content): {seed_authoritativeness}

Seed image manifest (images attached to this packet at creation; reference them in your output via `![alt](coco-img:img_<id>)`):
{image_manifest}

Source excerpts:
---
{seed_content}
---

Your task:
1. Distill the excerpts into clean markdown content.
2. Identify topic facets (1-3 to start) — distinct "ways someone might invoke this packet" (<=10 words each). Include the seed topic as one facet if it still applies.
3. Identify entities — proper nouns, names, places, named non-common nouns. Lowercase them. Exclude generic words like "family", "wife", "week", "day".
4. The seed content already contains `coco-img:img_<id>` references for any images attached. Preserve the ids verbatim; you may drop a reference if the image is purely decorative, but do not invent new ids.
5. Phrase the gist and summary with appropriate hedging when seed_authoritativeness is low (≤ 0.3) — e.g. "Per the conversation, ..."; for high trust (≥ 0.8) state facts directly.

Output a single JSON object only (no prose, no fences):
{{
  "gist": "one-line summary (under 15 words)",
  "summary": "one-paragraph summary (under 100 words)",
  "full": "the complete content (markdown)",
  "topic_facets": ["facet1"],
  "entities": ["entity1"]
}}"""


def _format_packet_image_manifest_inline(images: list, indent: str = "    ") -> str:
    """One line per image: `[img_<id>] alt="..." NNKB mime WxH`. Empty if no images."""
    if not images:
        return ""
    lines = []
    for img in images:
        alt = (img.alt or "").replace("\n", " ").strip()
        kb = max(1, len(img.data_b64 or "") * 3 // 4 // 1024)
        w, h = (img.dimensions or [0, 0])[:2]
        dim = f"{w}x{h}" if w and h else "vector"
        mime_short = (img.mime or "").split("/")[-1]
        lines.append(f'{indent}[{img.id}] alt={alt!r} {kb}KB {mime_short} {dim}')
    return "\n".join(lines)


def format_packets_for_prompt(loaded_packets: list[dict]) -> str:
    if not loaded_packets:
        return "(none)"
    parts = []
    for item in loaded_packets:
        packet = item["packet"]
        slice_type = item["slice"]
        content = getattr(packet.content, slice_type)
        facets_str = ", ".join(f'"{t.text}"' for t in packet.topics)
        ents_str = ", ".join(packet.entities[:20])
        src_urls_fn = getattr(packet, "source_urls", None)
        src_urls = src_urls_fn() if callable(src_urls_fn) else (src_urls_fn or [])
        src_line = ""
        if src_urls:
            src_line = f"  source_urls: {', '.join(src_urls[:5])}\n"
        auth_val = float(getattr(packet, "authoritativeness", 0.0) or 0.0)
        auth_line = f"  authoritativeness: {auth_val:.2f}\n" if auth_val > 0 else ""
        images = getattr(packet, "images", []) or []
        # Image manifest shown only for full-slice packets: those are the ones
        # whose bytes get attached as multimodal blocks (see build_user_content_blocks).
        img_line = ""
        if images and slice_type == "full":
            img_line = (
                "  images (correlate with the image content blocks attached to "
                "the user message):\n"
                f"{_format_packet_image_manifest_inline(images, indent='    ')}\n"
            )
        parts.append(
            f"--- packet id: {packet.id} | slice: {slice_type} ---\n"
            f"  facets: [{facets_str}]\n"
            f"  entities: [{ents_str}]\n"
            f"{src_line}"
            f"{auth_line}"
            f"{img_line}"
            f"  content:\n{content}"
        )
    return "\n\n".join(parts)


def build_system_prompt(
    loaded_packets: list[dict],
    user_name: str = "the user",
    ingest_mode: bool = False,
    upload_mode: bool = False,
) -> str:
    base = SYSTEM_PROMPT_TEMPLATE.format(
        user_name=user_name,
        loaded_packets=format_packets_for_prompt(loaded_packets),
    )
    if upload_mode:
        return base + DOCUMENT_SYSTEM_ADDENDUM
    if ingest_mode:
        return base + INGEST_SYSTEM_ADDENDUM
    return base


def build_document_batch_user_blocks(
    metadata_dict: dict,
    chunks: list,
    batch_index: int,
    total_pages_seen: int,
) -> list[dict]:
    """Frame a document batch for the main-reply LLM.

    metadata_dict: subset of DocumentMetadata rendered as text (filename,
                   format, document_type, file_authoritativeness).
    chunks:        list of DocumentChunk from documents.py.

    Returns a list-of-content-blocks (single text block; no images in v1).
    """
    lines: list[str] = []
    lines.append("<document>")
    lines.append(f"  filename: {metadata_dict.get('filename', '')}")
    lines.append(f"  format: {metadata_dict.get('format', '')}")
    lines.append(f"  document_type: {metadata_dict.get('document_type', '')}")
    fa = metadata_dict.get("file_authoritativeness")
    if fa is not None:
        lines.append(f"  file_authoritativeness: {float(fa):.2f}")
    lines.append("</document>")
    lines.append("")
    lines.append(f"Batch #{batch_index} (cumulative pages seen: {total_pages_seen})")
    lines.append("")
    for c in chunks:
        cref = c.chunk_ref()
        # Only include the paragraph attr when it applies.
        if c.paragraph_index is None:
            attrs = f'id="{cref}" page="{c.page_number}"'
        else:
            attrs = (
                f'id="{cref}" page="{c.page_number}" '
                f'paragraph="{c.paragraph_index}"'
            )
        lines.append(f"<chunk {attrs}>")
        lines.append(c.text)
        lines.append("</chunk>")
        lines.append("")

    return [{"type": "text", "text": "\n".join(lines)}]


def _format_fetch_image_manifest(images: dict) -> str:
    """One line per fetch-candidate image, keyed by [IMG_n]."""
    if not images:
        return "(no images available)"
    lines = []
    for key in sorted(images.keys()):
        blob = images[key]
        alt = (blob.alt or "").strip()
        w, h = blob.dimensions
        kb = max(1, blob.post_downscale_bytes // 1024)
        dim_str = f"{w}x{h}" if w and h else "vector"
        lines.append(
            f"[{key}] alt={alt!r} {kb}KB {blob.mime.split('/')[-1]} {dim_str}"
        )
    return "\n".join(lines)


def _format_sources_block(fetch_results: list) -> str:
    """Render <fetched_sources><source><markdown>...<images>...</source></fetched_sources>."""
    if not fetch_results:
        return ""
    parts = ["<fetched_sources>"]
    for fr in fetch_results:
        title_attr = f' title="{fr.title}"' if fr.title else ""
        truncated_attr = ' truncated="true"' if fr.truncated else ""
        parts.append(f'  <source url="{fr.url}"{title_attr}{truncated_attr}>')
        parts.append("    <markdown>")
        parts.append(fr.markdown)
        parts.append("    </markdown>")
        parts.append("    <images>")
        parts.append(_format_fetch_image_manifest(fr.images))
        parts.append("    </images>")
        parts.append("  </source>")
    parts.append("</fetched_sources>")
    return "\n".join(parts)


# MIME types Anthropic accepts as `image` content blocks.
_MULTIMODAL_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _make_image_block(media_type: str, data_b64: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data_b64},
    }


def build_user_content_blocks(
    history: list[dict],
    current_message: str,
    loaded_packets: list[dict] | None = None,
    fetch_results: list | None = None,
    failed_urls: list[tuple[str, str]] | None = None,
    image_blocks_max: int | None = None,
) -> list[dict]:
    """Build a multimodal user-content list: one text block followed by image blocks.

    Image blocks come from:
      - loaded_packets at slice='full' (each PacketImage → one image block)
      - fetch_results (each non-SVG ImageBlob → one image block)

    If `image_blocks_max` is set and the total would exceed it, packet images
    are dropped first (FIFO over the loaded_packets order), then fetch images.
    The text block always reports the alt manifest of dropped images too, so
    the LLM still knows they exist.
    """
    # --- text block ----------------------------------------------------------
    lines: list[str] = []
    if history:
        lines.append("Recent conversation:")
        for turn in history:
            role = "User" if turn["role"] == "user" else "Coco"
            lines.append(f"{role}: {turn['content']}")
        lines.append("")
    lines.append(f"User now says: {current_message}")
    lines.append("")
    if failed_urls:
        lines.append("URLs that could not be fetched (mention briefly in your reply):")
        for url, reason in failed_urls:
            lines.append(f"  - {url}: {reason}")
        lines.append("")
    if fetch_results:
        lines.append(_format_sources_block(fetch_results))

    text_block = {"type": "text", "text": "\n".join(lines)}

    # --- image blocks --------------------------------------------------------
    # Collect candidates in priority order: fetch (decision-critical) first,
    # then loaded-packet images. Trim packet images when over cap.
    fetch_image_blocks: list[dict] = []
    if fetch_results:
        for fr in fetch_results:
            for key in sorted(fr.images.keys()):
                blob = fr.images[key]
                if blob.mime in _MULTIMODAL_MIMES:
                    fetch_image_blocks.append(_make_image_block(blob.mime, blob.data_b64))

    packet_image_blocks: list[dict] = []
    packet_image_drops: list[str] = []
    if loaded_packets:
        for item in loaded_packets:
            if item.get("slice") != "full":
                continue
            packet = item["packet"]
            for img in getattr(packet, "images", []) or []:
                if img.mime in _MULTIMODAL_MIMES:
                    packet_image_blocks.append(_make_image_block(img.mime, img.data_b64))

    if image_blocks_max is not None and image_blocks_max >= 0:
        # Fetch images stay; trim packet images from the end if over cap.
        budget_for_packets = max(0, image_blocks_max - len(fetch_image_blocks))
        if len(packet_image_blocks) > budget_for_packets:
            packet_image_drops = [
                f"(dropped {len(packet_image_blocks) - budget_for_packets} loaded-packet "
                "image block(s) due to per-turn cap)"
            ]
            packet_image_blocks = packet_image_blocks[:budget_for_packets]
        # Final hard cap (in case fetch images themselves blew past the limit)
        all_blocks = fetch_image_blocks + packet_image_blocks
        if len(all_blocks) > image_blocks_max:
            all_blocks = all_blocks[:image_blocks_max]
    else:
        all_blocks = fetch_image_blocks + packet_image_blocks

    if packet_image_drops:
        text_block["text"] += "\n" + "\n".join(packet_image_drops)

    return [text_block] + all_blocks


def render_integrate_prompt(
    existing_facets: str,
    existing_entities: str,
    existing_content: str,
    existing_source_urls: list[str] | None,
    existing_authoritativeness: float,
    new_content: str,
    new_source_url: str | None,
    new_authoritativeness: float,
    writer_role: str,
    image_manifest_items: list | None = None,
) -> str:
    src_clause = (
        f" (newly ingested from {new_source_url})"
        if new_source_url
        else ""
    )
    urls_str = ", ".join(existing_source_urls or []) if existing_source_urls else "(none)"
    if image_manifest_items:
        manifest_text = _format_packet_image_manifest_inline(image_manifest_items, indent="    ")
    else:
        manifest_text = "    (no images)"
    return INTEGRATE_PROMPT.format(
        existing_facets=existing_facets,
        existing_entities=existing_entities,
        existing_source_urls=urls_str,
        existing_authoritativeness=f"{float(existing_authoritativeness):.2f}",
        image_manifest=manifest_text,
        existing_content=existing_content,
        new_content=new_content,
        new_source_url_clause=src_clause,
        new_authoritativeness=f"{float(new_authoritativeness):.2f}",
        writer_role=writer_role or "(unknown)",
    )


def render_new_packet_prompt(
    seed_topic: str,
    seed_content: str,
    seed_source_url: str | None,
    seed_authoritativeness: float,
    writer_role: str,
    image_manifest_items: list | None = None,
) -> str:
    clause = (
        f'Source URL (newly ingested): {seed_source_url}\n'
        if seed_source_url
        else ""
    )
    if image_manifest_items:
        manifest_text = _format_packet_image_manifest_inline(image_manifest_items, indent="    ")
    else:
        manifest_text = "    (no images)"
    return NEW_PACKET_PROMPT.format(
        seed_topic=seed_topic,
        seed_content=seed_content,
        seed_source_url_clause=clause,
        seed_authoritativeness=f"{float(seed_authoritativeness):.2f}",
        writer_role=writer_role or "(unknown)",
        image_manifest=manifest_text,
    )


def extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


_REPLY_RE = re.compile(r"<reply>(.*?)</reply>", re.DOTALL)
_METADATA_RE = re.compile(r"<metadata>(.*?)</metadata>", re.DOTALL)


def parse_coco_response(raw: str) -> dict:
    """Extracts <reply> body and parses <metadata> JSON."""
    reply_match = _REPLY_RE.search(raw)
    metadata_match = _METADATA_RE.search(raw)

    reply = reply_match.group(1).strip() if reply_match else raw.strip()

    packets_used: list = []
    new_knowledge: list = []
    if metadata_match:
        try:
            metadata_text = extract_json_block(metadata_match.group(1))
            md = json.loads(metadata_text)
            packets_used = md.get("packets_used", []) or []
            new_knowledge = md.get("new_knowledge", []) or []
        except Exception:
            # Metadata couldn't be parsed; reply still usable
            pass

    return {
        "reply": reply,
        "packets_used": packets_used,
        "new_knowledge": new_knowledge,
    }


_VALID_TRUST_RESOLUTIONS = {"new_wins", "existing_wins", "equal_escalate"}


def parse_integration_response(raw: str) -> dict:
    d = json.loads(extract_json_block(raw))
    # Normalize trust_resolution: must be one of the three documented values.
    # When the LLM omits it, fall back to a sensible default that preserves
    # the legacy "conflict → ask user" behaviour without crashing the path.
    tr = d.get("trust_resolution")
    if tr not in _VALID_TRUST_RESOLUTIONS:
        tr = "equal_escalate" if d.get("conflict_detected") else "new_wins"
    d["trust_resolution"] = tr
    # When trust_resolution is anything except equal_escalate, conflict_detected
    # must be false (the LLM is told this in the prompt, but a defensive clamp
    # avoids accidental user prompts on auto-resolved merges).
    if tr != "equal_escalate":
        d["conflict_detected"] = False
    else:
        d["conflict_detected"] = bool(d.get("conflict_detected", True))
    return d


def parse_new_packet_response(raw: str) -> dict:
    return json.loads(extract_json_block(raw))


def parse_extraction_response(raw: str) -> dict:
    return json.loads(extract_json_block(raw))
