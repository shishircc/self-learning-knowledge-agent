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


def banner_admin_warning() -> None:
    """Unmissable warning block for local admin (--admin) mode.

    Rendered at session start before banner_welcome. Communicates three things
    the user must not lose sight of:
      1. The session is UNAUTHENTICATED.
      2. Every mutation runs with full admin trust.
      3. This mode must not be used in production.
    """
    box_top = "╔══════════════════════════════════════════════════════════════════╗"
    box_bot = "╚══════════════════════════════════════════════════════════════════╝"
    lines = [
        "║  ⚠  LOCAL ADMIN MODE — UNAUTHENTICATED                           ║",
        "║     Full admin capabilities, no SSO.                             ║",
        "║     Unadvisable outside local development.                       ║",
        "║     Every mutation runs with admin trust.                        ║",
        "║     Do NOT use in production.                                    ║",
    ]
    _styled("")
    _styled(f"<b><ansired>{box_top}</ansired></b>")
    for ln in lines:
        _styled(f"<b><ansired>{ln}</ansired></b>")
    _styled(f"<b><ansired>{box_bot}</ansired></b>")
    _styled("")


def banner_goodbye(admin_mode: bool = False) -> None:
    _styled("")
    _styled("<ansicyan>Goodbye. Talk soon.</ansicyan>")
    if admin_mode:
        _styled(
            "<i><ansired>local admin mode — session was unauthenticated</ansired></i>"
        )
    _styled("")


def coco_label(admin_mode: bool = False) -> None:
    """Print 'Coco:' label (no newline) before the streamed reply.

    In admin mode, appends a dim red `(admin mode)` marker on the same line
    so the badge survives long conversations where the startup banner has
    scrolled off.
    """
    if admin_mode:
        _styled(
            "\n<b><ansicyan>Coco:</ansicyan></b>"
            " <i><ansired>(admin mode)</ansired></i> ",
            end="",
        )
    else:
        _styled("\n<b><ansicyan>Coco:</ansicyan></b> ", end="")


def hint(text: str) -> None:
    """A subtle dim hint line (e.g. recalling memory)."""
    _styled(f"<i><ansibrightblack>{text}</ansibrightblack></i>")


def user_prompt_html(admin_mode: bool = False) -> str:
    """HTML string to pass as the prompt label to PromptSession.prompt_async.

    In admin mode, prepends a bold red [ADMIN] badge so the user cannot type a
    message without seeing the mode on the same line as their input.
    """
    if admin_mode:
        return "<b><ansired>[ADMIN]</ansired></b> <b><ansigreen>You:</ansigreen></b> "
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
