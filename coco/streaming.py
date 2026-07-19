"""Keystroke-streaming console reader. Yields StreamEvent objects.

Built on prompt_toolkit's PromptSession + Buffer.on_text_changed hook. When the
terminal can't be put in raw mode (e.g. inside a non-TTY environment), falls
back to blocking line input — `partial` events never fire and each line becomes
a single `submit`.
"""
import asyncio
import sys
from dataclasses import dataclass

try:
    from prompt_toolkit import HTML, PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    HAVE_PT = True
except ImportError:
    HAVE_PT = False
    patch_stdout = None  # type: ignore
    HTML = None  # type: ignore

from . import ui


@dataclass
class StreamEvent:
    kind: str  # "partial" | "submit" | "cancel"
    text: str


def _count_words(text: str) -> int:
    return len(text.split())


class _PartialDispatcher:
    """Tracks buffer state and decides when to fire `partial` events."""

    def __init__(self, config: dict, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.config = config
        self.queue = queue
        self.loop = loop
        self.last_partial_text = ""
        self.words_at_last_partial = 0
        self.partials_fired = 0
        self.debounce_task: asyncio.Task | None = None

    def reset(self):
        self.last_partial_text = ""
        self.words_at_last_partial = 0
        self.partials_fired = 0
        if self.debounce_task and not self.debounce_task.done():
            self.debounce_task.cancel()
        self.debounce_task = None

    async def _fire(self, text: str):
        if self.partials_fired >= self.config["streaming_max_partials_per_turn"]:
            return
        if len(text) < self.config["streaming_min_chars"]:
            return
        if text == self.last_partial_text:
            return
        self.last_partial_text = text
        self.words_at_last_partial = _count_words(text)
        self.partials_fired += 1
        await self.queue.put(StreamEvent("partial", text))

    async def _debounce_fire(self, text: str):
        try:
            await asyncio.sleep(self.config["streaming_debounce_ms"] / 1000.0)
            await self._fire(text)
        except asyncio.CancelledError:
            pass

    def on_text_changed(self, text: str):
        # Called from prompt_toolkit's event loop; we're in async context.
        if self.debounce_task and not self.debounce_task.done():
            self.debounce_task.cancel()

        current_words = _count_words(text)
        words_since = current_words - self.words_at_last_partial
        if (
            words_since >= self.config["streaming_words_per_partial"]
            and len(text) >= self.config["streaming_min_chars"]
        ):
            asyncio.run_coroutine_threadsafe(self._fire(text), self.loop)
        else:
            self.debounce_task = self.loop.create_task(self._debounce_fire(text))


async def _fallback_input_stream(config: dict, admin_mode: bool = False):
    """Blocking input — no partial events."""
    label = "[ADMIN] You: " if admin_mode else "You: "
    while True:
        try:
            text = await asyncio.to_thread(input, label)
        except (EOFError, KeyboardInterrupt):
            yield StreamEvent("cancel", "")
            return
        text = text.strip()
        if text.lower() in ("exit", "quit"):
            yield StreamEvent("cancel", "")
            return
        if text:
            yield StreamEvent("submit", text)


async def input_stream(config: dict, admin_mode: bool = False):
    """Async generator over StreamEvents from the user's typing.

    `admin_mode` (when true) tells the UI to render a bold red [ADMIN] badge
    on every prompt so the user cannot type without seeing the mode.
    """
    if not HAVE_PT or not sys.stdin.isatty():
        async for ev in _fallback_input_stream(config, admin_mode=admin_mode):
            yield ev
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    dispatcher = _PartialDispatcher(config, queue, loop)
    session = PromptSession()

    def on_text_changed(buf):
        dispatcher.on_text_changed(buf.text)

    # prompt_toolkit Event supports __iadd__ for handler registration
    session.default_buffer.on_text_changed += on_text_changed

    while True:
        dispatcher.reset()

        async def read_submit():
            try:
                with patch_stdout(raw=True):
                    text = await session.prompt_async(
                        HTML(ui.user_prompt_html(admin_mode=admin_mode))
                    )
                if text.strip().lower() in ("exit", "quit"):
                    await queue.put(StreamEvent("cancel", ""))
                else:
                    await queue.put(StreamEvent("submit", text))
            except (EOFError, KeyboardInterrupt):
                await queue.put(StreamEvent("cancel", ""))

        read_task = asyncio.create_task(read_submit())

        last_event_kind = None
        while True:
            event = await queue.get()
            yield event
            last_event_kind = event.kind
            if event.kind in ("submit", "cancel"):
                break

        try:
            await read_task
        except Exception:
            pass

        if last_event_kind == "cancel":
            break
