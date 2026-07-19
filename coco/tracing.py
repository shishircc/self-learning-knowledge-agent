"""Langfuse tracing wrapper.

No-op when EITHER:
  - config["tracing"]["enabled"] is false (config wins; env is not consulted), OR
  - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are absent from the environment.

See TDS.md §2.8 and §6.4 for the precedence rules and rationale.
"""
import os
from contextlib import contextmanager

_client = None
_enabled = False
_initialized = False


def _config_gate(config) -> bool:
    """Read config["tracing"]["enabled"] tolerantly.

    Missing / malformed → default true (env-only gate as before).
    Truthy non-bool → true; falsy non-bool → false.
    """
    if not isinstance(config, dict):
        return True
    section = config.get("tracing")
    if not isinstance(section, dict):
        return True
    if "enabled" not in section:
        return True
    return bool(section["enabled"])


def init(config=None) -> bool:
    """Idempotent. Returns True if tracing is active.

    Gate order:
      1. config["tracing"]["enabled"] must be truthy (default true when absent).
         When false: short-circuit before importing `langfuse` or reading env.
      2. LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY must be present in env.
      3. `langfuse.Langfuse(...)` must instantiate cleanly.
    """
    global _client, _enabled, _initialized
    if _initialized:
        return _enabled
    _initialized = True

    if not _config_gate(config):
        _enabled = False
        return False

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
def session_context(
    session_id: str,
    user_id: str | None = None,
    metadata: dict | None = None,
):
    """Bind subsequent spans to a Langfuse session (and optional user + metadata).

    Metadata propagates as OTel baggage so all child spans pick it up; used to
    record role / role_authoritativeness / provider for trust-aware analysis.
    No-op when tracing is disabled.
    """
    if not _enabled:
        yield
        return
    from langfuse import propagate_attributes
    kwargs: dict = {"session_id": session_id}
    if user_id:
        kwargs["user_id"] = user_id
    if metadata:
        kwargs["metadata"] = metadata
    try:
        with propagate_attributes(**kwargs):
            yield
    except TypeError:
        # Older langfuse may not accept `metadata=`; fall back without it.
        with propagate_attributes(session_id=session_id, user_id=user_id):
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
