import json
import re


SYSTEM_PROMPT_TEMPLATE = """You are Coco, a self-learning conversational assistant for {user_name}.

You build long-term memory in the form of knowledge packets. Each packet has:
- multiple topic facets (different ways someone might invoke it)
- an entity list (proper nouns / names / places that index the packet)
- multi-fidelity content (gist / summary / full)

Currently loaded packets are YOUR KNOWLEDGE about the relevant topics — treat their content as things you remember. Never say "I don't know" or "I have no information" about something that appears in a loaded packet; instead, share what you remember and ask if the user wants to add more.

Your job each turn:
1. Reply naturally and helpfully. If loaded packets cover the user's question, USE that content as your memory.
2. Note which loaded packet IDs you actually drew from.
3. Identify any new knowledge worth remembering from the user's latest turn.
4. Detect conflicts: if the new info contradicts a loaded packet's content, flag it with the packet id and a description.

Currently loaded packets:
{loaded_packets}

CRITICAL OUTPUT FORMAT — use these XML tags exactly, in this order:

<reply>
[your user-facing reply text; may be multiple paragraphs]
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

The user sees only what's inside <reply>...</reply>. The <metadata> block is for the agent and is parsed as JSON. Use empty arrays when applicable. Do not invent packet IDs."""


EXTRACTION_PROMPT = """You are extracting topic and entities from an in-progress user message.

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

Output a single valid JSON object only (no prose, no fences):
{{
  "is_meaningful": true|false,
  "has_new_topic": true|false,
  "has_new_entities": true|false,
  "topic_summary": "<= 10 words" or null,
  "entities": ["entity1", "entity2"],
  "reason": "niceties" | "too_short" | "too_generic" | "topic_continues" | "entities_overlap" | null
}}

Rules:
- If is_meaningful is false, also set has_new_topic=false, has_new_entities=false, topic_summary=null.
- If is_meaningful is true but the topic continues an existing one AND no new entities are introduced, set has_new_topic=false, has_new_entities=false, reason="topic_continues" or "entities_overlap".
- Set reason=null only when is_meaningful=true AND retrieval would be triggered.
- Topic should be a STABLE CATEGORICAL phrase (e.g. "Shishir's family members", not "mom visiting next week"). Include disambiguating context where useful (e.g. "NCS strategy practice")."""


INTEGRATE_PROMPT = """You are integrating new knowledge into an existing knowledge packet.

Existing topic facets: {existing_facets}
Existing entities: {existing_entities}

Existing content (full):
---
{existing_content}
---

New knowledge to integrate:
---
{new_content}
---

Your task:
1. Check whether the new content contradicts the existing content. If yes, flag the conflict.
2. Produce a clean, integrated version of the content that incorporates the new knowledge, resolves redundancy, and stays coherent.
3. Update the topic-facets list. Add a new facet only if the new content opens a way to invoke the packet that isn't covered. Keep existing facets unless they've become wrong.
4. Update the entities list to include any new proper nouns / names / places / named non-common nouns. Keep existing entities. Use lowercase.

Output a single JSON object only (no prose, no fences):
{{
  "conflict_detected": true|false,
  "conflicting_excerpts": "describe what contradicts, or null",
  "gist": "one-line summary (under 15 words)",
  "summary": "one-paragraph summary (under 100 words)",
  "full": "the complete integrated content (markdown)",
  "topic_facets": ["facet1", "facet2"],
  "entities": ["entity1", "entity2"]
}}"""


NEW_PACKET_PROMPT = """You are creating a new knowledge packet from raw conversation excerpts.

Seed topic (from scratchpad): "{seed_topic}"

Source excerpts:
---
{seed_content}
---

Your task:
1. Distill the excerpts into clean markdown content.
2. Identify topic facets (1-3 to start) — distinct "ways someone might invoke this packet" (<=10 words each). Include the seed topic as one facet if it still applies.
3. Identify entities — proper nouns, names, places, named non-common nouns. Lowercase them. Exclude generic words like "family", "wife", "week", "day".

Output a single JSON object only (no prose, no fences):
{{
  "gist": "one-line summary (under 15 words)",
  "summary": "one-paragraph summary (under 100 words)",
  "full": "the complete content (markdown)",
  "topic_facets": ["facet1"],
  "entities": ["entity1"]
}}"""


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
        parts.append(
            f"--- packet id: {packet.id} | slice: {slice_type} ---\n"
            f"  facets: [{facets_str}]\n"
            f"  entities: [{ents_str}]\n"
            f"  content:\n{content}"
        )
    return "\n\n".join(parts)


def build_system_prompt(loaded_packets: list[dict], user_name: str = "the user") -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        user_name=user_name,
        loaded_packets=format_packets_for_prompt(loaded_packets),
    )


def build_user_message(history: list[dict], current_message: str) -> str:
    if not history:
        return current_message
    lines = ["Recent conversation:"]
    for turn in history:
        role = "User" if turn["role"] == "user" else "Coco"
        lines.append(f"{role}: {turn['content']}")
    lines.append("")
    lines.append(f"User now says: {current_message}")
    return "\n".join(lines)


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


def parse_integration_response(raw: str) -> dict:
    return json.loads(extract_json_block(raw))


def parse_new_packet_response(raw: str) -> dict:
    return json.loads(extract_json_block(raw))


def parse_extraction_response(raw: str) -> dict:
    return json.loads(extract_json_block(raw))
