import math
from datetime import datetime, timezone


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def compute_strength(events: list, weights: dict, half_life_days: float, now=None) -> float:
    """Sum decayed weights of all events."""
    if not events:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)

    half_life_seconds = half_life_days * 86400
    decay_constant = math.log(2) / half_life_seconds

    total = 0.0
    for event in events:
        weight = event.get("weight", weights.get(event["event_type"], 1.0))
        delta = (now - parse_iso(event["timestamp"])).total_seconds()
        total += weight * math.exp(-decay_constant * delta)
    return total


def slice_for_strength(strength: float, band_gist_max: float, band_summary_max: float) -> str:
    if strength < band_gist_max:
        return "gist"
    if strength < band_summary_max:
        return "summary"
    return "full"


def strength_bias(strength: float, scale: float) -> float:
    return scale * math.log1p(strength)
