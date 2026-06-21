"""Langfuse tracing wrapper. No-op when LANGFUSE_* env vars are absent."""
import os
from contextlib import contextmanager

_client = None
_enabled = False
_initialized = False


def init() -> bool:
    """Idempotent. Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.
    Returns True if tracing is active.
    """
    global _client, _enabled, _initialized
    if _initialized:
        return _enabled
    _initialized = True

    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not (public and secret):
        _enabled = False
        return False

    try:
        from langfuse import Langfuse
        _client = Langfuse(public_key=public, secret_key=secret, host=host)
        _enabled = True
        return True
    except Exception as e:
        print(f"[langfuse init failed: {e}]")
        _enabled = False
        return False


def enabled() -> bool:
    return _enabled


@contextmanager
def observation(name: str, as_type: str = "span", input=None, metadata=None, model=None):
    """Context manager: opens a Langfuse span/generation/etc. No-op if disabled."""
    if not _enabled:
        yield None
        return
    kwargs = {"name": name, "as_type": as_type}
    if input is not None:
        kwargs["input"] = input
    if metadata is not None:
        kwargs["metadata"] = metadata
    if model is not None and as_type == "generation":
        kwargs["model"] = model
    with _client.start_as_current_observation(**kwargs) as obs:
        yield obs


@contextmanager
def session_context(session_id: str, user_id: str | None = None):
    """Bind subsequent spans to a Langfuse session (and optional user). No-op if disabled."""
    if not _enabled:
        yield
        return
    from langfuse import propagate_attributes
    kwargs = {"session_id": session_id}
    if user_id:
        kwargs["user_id"] = user_id
    with propagate_attributes(**kwargs):
        yield


def update(obs, **kwargs):
    """Safe update on a Langfuse observation; no-op if obs is None."""
    if obs is None:
        return
    try:
        obs.update(**kwargs)
    except Exception:
        pass


def score(name: str, value, comment: str | None = None):
    if not _enabled:
        return
    try:
        _client.score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        pass


def flush():
    if _enabled and _client is not None:
        try:
            _client.flush()
        except Exception:
            pass
