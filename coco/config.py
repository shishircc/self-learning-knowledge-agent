import json
from pathlib import Path

DEFAULT_CONFIG = {
    "topic_match_threshold": 0.50,
    "retrieval_threshold": 0.50,
    "existing_packet_match_threshold": 0.60,
    "facet_dedup_threshold": 0.85,
    "scratchpad_promote_threshold": 0.65,
    "recency_window": 5,
    "hybrid_search_method": "RRF",
    "hybrid_search_k": 2,
    "hybrid_search_weights": {"bm25": 0.5, "cosine": 0.5},
    "cosine_channel_floor": 0.1,
    "strength_weights": {"retrieval": 1, "use": 3, "write": 5},
    "strength_half_life_days": 30,
    "strength_additive_bias_scale": 0.002,
    "band_gist_max": 5.0,
    "band_summary_max": 15.0,
    "scratchpad_discard_after_sessions": 10,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "anthropic_model": "claude-sonnet-4-5",
    "small_lm_model": "claude-haiku-4-5",
    "data_dir": "./data",
    "streaming_debounce_ms": 350,
    "streaming_words_per_partial": 5,
    "streaming_min_chars": 12,
    "streaming_max_partials_per_turn": 8,
    "ingest_enabled": True,
    "ingest_user_agent": "coco/0.2 (+https://github.com/shishir/self-learning-knowledge-agent)",
    "ingest_request_timeout_s": 15,
    "ingest_max_page_bytes": 5_000_000,
    "ingest_min_article_chars": 200,
    "ingest_markdown_max_chars": 120_000,
    "ingest_image_max_dim": 1280,
    "ingest_image_max_bytes": 500_000,
    "ingest_image_concurrency": 4,
    "ingest_max_images_per_page": 20,
    "ingest_allowed_image_mimes": [
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
    ],
    "image_blocks_max_per_turn": 20,
    "debug_print_state": False,
    "debug_print_streaming": False,
    "debug_print_write_path": False,
    # ---- Identity / auth ----
    "auth": {
        "startup_mode": "prompt",
        "providers": ["anonymous", "entra", "google"],
        "default_provider": None,
        "default_role": "user",
        "fallback_to_anonymous": True,
        "entra": {
            "tenant_id": None,
            "client_id": None,
            "scopes": ["openid", "profile", "email", "User.Read"],
            "flow": "device_code",
        },
        "google": {
            "client_id": None,
            "scopes": ["openid", "profile", "email"],
            "redirect_uri": "http://localhost:53682/callback",
        },
        "email_role_map": {},
        "entra_group_role_map": {},
        "role_capabilities": {},  # empty => DEFAULT_ROLE_CAPABILITIES used per role
    },
    # ---- Source trust ----
    "domain_authoritativeness": {},
    "default_domain_authoritativeness": 0.5,
    "authoritativeness_bias_scale": 0.001,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Nested dicts are merged key-by-key;
    other values (including lists) are replaced wholesale by the override.
    Returns a new dict; does not mutate `base`.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_CONFIG)
    with open(p) as f:
        user_config = json.load(f)
    return _deep_merge(DEFAULT_CONFIG, user_config)
