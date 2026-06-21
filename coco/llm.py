"""Direct Anthropic SDK wrapper. Lazy client init; picks up ANTHROPIC_API_KEY from env."""
from anthropic import AsyncAnthropic

_client: AsyncAnthropic | None = None


def anthropic_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


DEFAULT_MAX_TOKENS = 4096
