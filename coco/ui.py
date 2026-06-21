"""End-user-facing console output: banners, prompts, and Coco's label.

Uses prompt_toolkit's HTML formatting for ANSI colors and styles. All
developer-oriented diagnostic output (packet IDs, scores, topics, entities,
state dumps) lives behind `debug_print_*` flags in `agent.py` and does NOT
go through this module.
"""
import sys

try:
    from prompt_toolkit import HTML, print_formatted_text
    HAVE_PT = True
except ImportError:
    HAVE_PT = False
    HTML = None  # type: ignore


def _plain(text: str, end: str = "\n") -> None:
    sys.stdout.write(text + end)
    sys.stdout.flush()


def _styled(html: str, end: str = "\n") -> None:
    if HAVE_PT:
        print_formatted_text(HTML(html), end=end)
    else:
        # Strip tags as a fallback
        import re
        text = re.sub(r"</?[a-zA-Z]+>", "", html)
        _plain(text, end=end)


def banner_welcome(user_name: str) -> None:
    _styled("")
    _styled("<ansicyan>──────────────────────────────────────────</ansicyan>")
    _styled(
        f"  <b><ansicyan>Coco</ansicyan></b>"
        f" — your conversational companion"
    )
    _styled(
        f"  <ansibrightblack>Memory grows from each conversation we have.</ansibrightblack>"
    )
    _styled("<ansicyan>──────────────────────────────────────────</ansicyan>")
    _styled(f"<ansibrightblack>Type your message below. Use </ansibrightblack>"
            f"<b>exit</b><ansibrightblack> or Ctrl-D to end.</ansibrightblack>")
    _styled("")


def banner_goodbye() -> None:
    _styled("")
    _styled("<ansicyan>Goodbye. Talk soon.</ansicyan>")
    _styled("")


def coco_label() -> None:
    """Print 'Coco:' label (no newline) before the streamed reply."""
    _styled("\n<b><ansicyan>Coco:</ansicyan></b> ", end="")


def hint(text: str) -> None:
    """A subtle dim hint line (e.g. recalling memory)."""
    _styled(f"<i><ansibrightblack>{text}</ansibrightblack></i>")


def user_prompt_html() -> str:
    """HTML string to pass as the prompt label to PromptSession.prompt_async."""
    return "<b><ansigreen>You:</ansigreen></b> "


def error(text: str) -> None:
    _styled(f"<ansired>{text}</ansired>")


# ---------------------------------------------------------------------------
# Brief memory-activity hints (always shown — load / save / update of packets)
# ---------------------------------------------------------------------------

_GIST_MAX_CHARS = 90


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _brief_gist(gist: str) -> str:
    snippet = (gist or "").strip().replace("\n", " ")
    if not snippet:
        return ""
    if len(snippet) > _GIST_MAX_CHARS:
        snippet = snippet[: _GIST_MAX_CHARS - 3].rstrip() + "..."
    return snippet


def memory_recall(gist: str) -> None:
    """Brief hint when a packet is loaded into context."""
    snippet = _brief_gist(gist)
    if not snippet:
        return
    _styled(
        f"<i><ansibrightblack>  recalling: {_escape(snippet)}</ansibrightblack></i>"
    )


def memory_saved(gist: str) -> None:
    """Brief hint when a new packet is committed to long-term memory."""
    snippet = _brief_gist(gist)
    if not snippet:
        return
    _styled(
        f"<i><ansibrightblack>  remembered: {_escape(snippet)}</ansibrightblack></i>"
    )


def memory_updated(gist: str) -> None:
    """Brief hint when an existing packet has new content integrated."""
    snippet = _brief_gist(gist)
    if not snippet:
        return
    _styled(
        f"<i><ansibrightblack>  updated: {_escape(snippet)}</ansibrightblack></i>"
    )
