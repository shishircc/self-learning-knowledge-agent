"""Identity acquisition + role / capability / source-trust resolution.

Public surface used by the rest of coco:
  - Identity                       dataclass (frozen)
  - ANONYMOUS / make_anonymous     identity for the no-login mode
  - ROLE_AUTHORITATIVENESS         role string -> trust scalar in [0, 1]
  - CAPABILITIES                   the catalogue of known capability strings
  - DEFAULT_ROLE_CAPABILITIES      fallback role -> capability set
  - acquire_identity(config)       drive the configured startup flow
  - login_entra / login_google     individual SSO entry points
  - resolve_role / resolve_capabilities
  - resolve_domain_authoritativeness
  - effective_authoritativeness    max(role_auth, domain_auth or 0)
  - CapabilityDenied / AuthError   exceptions
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Iterable
from fnmatch import fnmatch
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_AUTHORITATIVENESS: dict[str, float] = {
    "admin":     1.0,
    "author":    0.8,
    "viewer":    0.5,
    "user":      0.3,
    "anonymous": 0.0,
}

CAPABILITIES: frozenset[str] = frozenset({
    "read_packets",
    "write_scratchpad",
    "promote_scratchpad",
    "create_packet",
    "integrate_packet",
    "override_conflict",
    "delete_packet",
    "force_rewrite",
    "skill.fetch_url",
    "skill.upload_document",
})

DEFAULT_ROLE_CAPABILITIES: dict[str, frozenset[str]] = {
    "admin": frozenset({
        "read_packets", "write_scratchpad", "promote_scratchpad",
        "create_packet", "integrate_packet", "override_conflict",
        "delete_packet", "force_rewrite",
        "skill.fetch_url", "skill.upload_document",
    }),
    "author": frozenset({
        "read_packets", "write_scratchpad", "promote_scratchpad",
        "create_packet", "integrate_packet",
        "skill.fetch_url", "skill.upload_document",
    }),
    "viewer": frozenset({"read_packets"}),
    "user": frozenset({
        "read_packets", "write_scratchpad",
        "skill.fetch_url", "skill.upload_document",
    }),
    "anonymous": frozenset({"read_packets"}),
}

_DEFAULT_ROLE_FALLBACK = "user"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Raised when identity acquisition fails fatally (no fallback path)."""


class CapabilityDenied(Exception):
    """Raised by gated operations when the writer lacks a required capability."""

    def __init__(self, capability: str, role: str):
        super().__init__(f"role={role!r} lacks capability {capability!r}")
        self.capability = capability
        self.role = role


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Identity:
    name: str
    email: str | None
    role: str
    role_authoritativeness: float
    capabilities: frozenset[str]
    provider: str  # "anonymous" | "entra" | "google" | "cli_admin"
    claims: dict = field(default_factory=dict)

    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def prompt_display_name(self) -> str:
        # Name spliced into Coco's main-reply system prompt each turn so she
        # knows who she is speaking to from turn one. Anonymous mode degrades
        # to a generic label rather than surfacing the internal role string.
        if self.provider == "anonymous" or self.name == "anonymous":
            return "the user"
        return self.name

    def trace_metadata(self) -> dict:
        # Safe-for-trace projection; never includes raw claims (PII risk).
        md = {
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "role_authoritativeness": self.role_authoritativeness,
            "provider": self.provider,
        }
        # Local admin mode marker — surfaced so post-hoc Langfuse review can
        # filter unauthenticated dev sessions out of real-user traffic.
        if self.provider == "cli_admin":
            md["admin_mode"] = True
            md["unauthenticated"] = True
        return md

    @property
    def is_local_admin(self) -> bool:
        """True iff this identity was synthesized by the --admin CLI flag."""
        return self.provider == "cli_admin"


def _anonymous_capabilities(config: dict | None) -> frozenset[str]:
    if not config:
        return DEFAULT_ROLE_CAPABILITIES["anonymous"]
    role_caps_cfg = (config.get("auth") or {}).get("role_capabilities") or {}
    if "anonymous" in role_caps_cfg:
        return frozenset(c for c in role_caps_cfg["anonymous"] if c in CAPABILITIES)
    return DEFAULT_ROLE_CAPABILITIES["anonymous"]


def make_anonymous(config: dict | None = None) -> Identity:
    return Identity(
        name="anonymous",
        email=None,
        role="anonymous",
        role_authoritativeness=ROLE_AUTHORITATIVENESS["anonymous"],
        capabilities=_anonymous_capabilities(config),
        provider="anonymous",
        claims={},
    )


ANONYMOUS: Identity = make_anonymous(None)


# ---------------------------------------------------------------------------
# Local admin mode (--admin CLI flag)
# ---------------------------------------------------------------------------
#
# Developer escape hatch that bypasses SSO on startup. Concept, safety gates,
# and visual-signalling requirements live in DESIGN.md §"Local admin mode".
# This factory synthesizes a full-trust admin identity WITHOUT touching an
# IdP. Callers (only `acquire_identity`) must have already verified
# `config.auth.allow_cli_admin` — this function itself does not check it.

LOCAL_ADMIN_DEFAULT_NAME = "local-admin"


def local_admin_identity(config: dict, admin_name: str | None = None) -> Identity:
    """Synthesize the full-trust admin identity used by --admin mode.

    `name`     = admin_name or "local-admin"
    `email`    = None (no IdP round-trip)
    `role`     = "admin", role_authoritativeness = 1.0
    `provider` = "cli_admin"  — the marker that flags this session as
                                unauthenticated in Langfuse metadata and
                                (indirectly, via email=None) in per-write
                                PacketSource provenance.

    Capabilities come from the normal `resolve_capabilities("admin", config)`
    path so operator-configured tightening of the admin role still applies.
    """
    name = (admin_name or "").strip() or LOCAL_ADMIN_DEFAULT_NAME
    return Identity(
        name=name,
        email=None,
        role="admin",
        role_authoritativeness=ROLE_AUTHORITATIVENESS["admin"],
        capabilities=resolve_capabilities("admin", config),
        provider="cli_admin",
        claims={},
    )


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------

def _default_role(config: dict) -> str:
    role = (config.get("auth") or {}).get("default_role") or _DEFAULT_ROLE_FALLBACK
    if role not in ROLE_AUTHORITATIVENESS:
        return _DEFAULT_ROLE_FALLBACK
    return role


def lookup_role_for_email(email: str | None, config: dict) -> str:
    if not email:
        return _default_role(config)
    mapping = (config.get("auth") or {}).get("email_role_map") or {}
    e_lower = email.strip().lower()
    for k, v in mapping.items():
        if isinstance(k, str) and k.strip().lower() == e_lower:
            return v if v in ROLE_AUTHORITATIVENESS else _default_role(config)
    return _default_role(config)


def parse_role_from_entra_claims(claims: dict, config: dict) -> str | None:
    matched: list[str] = []

    # Entra App Roles: `roles: [str, ...]`
    for r in claims.get("roles") or []:
        if isinstance(r, str) and r in ROLE_AUTHORITATIVENESS:
            matched.append(r)

    # Group membership: `groups: [object_id, ...]`
    group_map = (config.get("auth") or {}).get("entra_group_role_map") or {}
    for g in claims.get("groups") or []:
        if isinstance(g, str) and g in group_map:
            r = group_map[g]
            if r in ROLE_AUTHORITATIVENESS:
                matched.append(r)

    if not matched:
        return None
    return max(matched, key=lambda r: ROLE_AUTHORITATIVENESS[r])


def resolve_role(
    email: str | None,
    claims: dict,
    provider: str,
    config: dict,
) -> tuple[str, float]:
    role: str | None = None
    if provider == "entra":
        role = parse_role_from_entra_claims(claims, config)
    if role is None:
        role = lookup_role_for_email(email, config)
    if role not in ROLE_AUTHORITATIVENESS:
        role = _default_role(config)
    return role, ROLE_AUTHORITATIVENESS[role]


def resolve_capabilities(role: str, config: dict) -> frozenset[str]:
    cfg = (config.get("auth") or {}).get("role_capabilities") or {}
    if role in cfg:
        return frozenset(c for c in (cfg[role] or []) if c in CAPABILITIES)
    return DEFAULT_ROLE_CAPABILITIES.get(role, frozenset())


# ---------------------------------------------------------------------------
# Source trust resolution
# ---------------------------------------------------------------------------

def _normalize_host(host: str) -> str:
    h = (host or "").strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h


def resolve_domain_authoritativeness(url: str | None, config: dict) -> float:
    """Longest-prefix match against config["domain_authoritativeness"].

    Keys are bare host ("en.wikipedia.org") or host + path-prefix
    ("internal.acme.com/handbook"). Host is matched case-insensitively
    after stripping a leading "www."; path is matched as an exact prefix.
    """
    default = float(config.get("default_domain_authoritativeness", 0.5))
    if not url:
        return default

    try:
        parsed = urlparse(url)
    except Exception:
        return default

    host = _normalize_host(parsed.hostname or "")
    path = parsed.path or ""
    if not host:
        return default

    mapping = config.get("domain_authoritativeness") or {}
    if not mapping:
        return default

    best_len = -1
    best_value: float | None = None
    for raw_key, raw_value in mapping.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key:
            continue
        key_host, _, key_path = key.partition("/")
        if _normalize_host(key_host) != host:
            continue
        key_path_norm = ("/" + key_path) if key_path else ""
        if key_path_norm and not path.startswith(key_path_norm):
            continue
        match_len = len(_normalize_host(key_host)) + len(key_path_norm)
        if match_len > best_len:
            best_len = match_len
            try:
                best_value = float(raw_value)
            except (TypeError, ValueError):
                best_value = None

    if best_value is None:
        return default
    return max(0.0, min(1.0, best_value))


def effective_authoritativeness(
    role_authoritativeness: float, domain_authoritativeness: float | None
) -> float:
    """The canonical max() rule: trust of a write = max(writer, source)."""
    role = float(role_authoritativeness or 0.0)
    domain = float(domain_authoritativeness or 0.0)
    return max(role, domain)


def resolve_file_authoritativeness(filename: str | None, config: dict) -> float:
    """Longest-match against config["file_authoritativeness"].

    Keys may be:
      - a filename glob   ("*.draft.pdf", "acme-handbook-*.pdf")
      - an absolute path prefix ("/policies/", "/Users/x/uploads/policies/")

    Resolution: longest key wins. Path prefixes beat globs on tie length.
    Path-prefix match is done against the absolute path (if `filename` is
    absolute); globs match the basename. Falls back to
    `default_file_authoritativeness` (default 0.5) when no key matches.
    """
    default = float(config.get("default_file_authoritativeness", 0.5))
    if not filename:
        return default

    mapping = config.get("file_authoritativeness") or {}
    if not mapping:
        return default

    basename = os.path.basename(filename)
    abs_path = filename if os.path.isabs(filename) else os.path.abspath(filename)

    best_len = -1
    best_is_prefix = False
    best_value: float | None = None

    for raw_key, raw_value in mapping.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key:
            continue

        is_prefix = key.startswith("/") and any(c not in "*?[]" for c in key)
        matched = False
        if is_prefix:
            if abs_path.startswith(key):
                matched = True
        else:
            if fnmatch(basename, key) or fnmatch(filename, key):
                matched = True

        if not matched:
            continue

        # Longest match wins; on equal length, path prefix beats bare glob.
        key_len = len(key)
        if (
            key_len > best_len
            or (key_len == best_len and is_prefix and not best_is_prefix)
        ):
            best_len = key_len
            best_is_prefix = is_prefix
            try:
                best_value = float(raw_value)
            except (TypeError, ValueError):
                best_value = None

    if best_value is None:
        return default
    return max(0.0, min(1.0, best_value))


# ---------------------------------------------------------------------------
# Identity acquisition at startup
# ---------------------------------------------------------------------------

async def acquire_identity(config: dict, cli_flags=None) -> Identity:
    """Drive the configured startup flow; always returns an Identity.

    `cli_flags` is the parsed argparse.Namespace from __main__ (or None in
    tests). When `cli_flags.admin` is truthy, the local-admin short-circuit
    fires BEFORE any other startup work:
      - config.auth.allow_cli_admin must be true, otherwise AuthError.
      - Returns local_admin_identity(config, cli_flags.admin_name) — no IdP,
        no prompt, no fallback.

    Modes (only reached when --admin is NOT set):
      "anonymous"     -> ANONYMOUS immediately.
      "authenticated" -> drive `default_provider` directly.
      "prompt"        -> interactive choice between the configured providers.
    """
    cfg = (config.get("auth") or {})

    # --- Local admin short-circuit ------------------------------------------
    if cli_flags is not None and getattr(cli_flags, "admin", False):
        if not bool(cfg.get("allow_cli_admin", False)):
            raise AuthError(
                "--admin is disabled in this config; "
                "set auth.allow_cli_admin=true in a local config to use it"
            )
        return local_admin_identity(config, getattr(cli_flags, "admin_name", None))

    mode = cfg.get("startup_mode", "prompt")
    providers = list(cfg.get("providers") or [])
    fallback = bool(cfg.get("fallback_to_anonymous", True))

    if mode == "anonymous" or not providers or providers == ["anonymous"]:
        return make_anonymous(config)

    if mode == "authenticated":
        choice = cfg.get("default_provider")
        if not choice:
            choice = next((p for p in providers if p != "anonymous"), None)
        if choice is None:
            return make_anonymous(config)
        try:
            return await _drive_provider(choice, config)
        except Exception as e:
            if fallback:
                _warn(f"login via {choice} failed ({e}); falling back to anonymous")
                return make_anonymous(config)
            raise AuthError(str(e)) from e

    # mode == "prompt"
    while True:
        try:
            choice = await _prompt_for_provider(providers)
        except (EOFError, KeyboardInterrupt):
            if fallback:
                return make_anonymous(config)
            raise AuthError("login cancelled") from None
        if choice == "anonymous":
            return make_anonymous(config)
        try:
            return await _drive_provider(choice, config)
        except Exception as e:
            _warn(f"login via {choice} failed: {e}")
            if not fallback:
                raise AuthError(str(e)) from e
            # else re-prompt


async def _drive_provider(choice: str, config: dict) -> Identity:
    cfg = (config.get("auth") or {})
    if choice == "entra":
        return await login_entra(cfg.get("entra") or {}, config)
    if choice == "google":
        return await login_google(cfg.get("google") or {}, config)
    if choice == "anonymous":
        return make_anonymous(config)
    raise AuthError(f"unknown provider {choice!r}")


async def _prompt_for_provider(providers: Iterable[str]) -> str:
    options = [p for p in providers if p in {"anonymous", "entra", "google"}]
    if not options:
        return "anonymous"

    labels = {"anonymous": "[a]nonymous", "entra": "[m]icrosoft", "google": "[g]oogle"}
    keymap = {
        "a": "anonymous", "anonymous": "anonymous",
        "m": "entra",     "microsoft": "entra", "entra": "entra",
        "g": "google",    "google": "google",
    }
    prompt = f"How would you like to sign in?  {' / '.join(labels[p] for p in options)}: "
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, lambda: input(prompt))
    raw = (raw or "").strip().lower()
    if raw in keymap and keymap[raw] in options:
        return keymap[raw]
    return options[0]


def _warn(msg: str) -> None:
    try:
        from . import ui
        ui.hint(f"auth: {msg}")
    except Exception:
        print(f"[auth] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Microsoft Entra (device-code flow via msal)
# ---------------------------------------------------------------------------

async def login_entra(provider_cfg: dict, config: dict) -> Identity:
    tenant_id = provider_cfg.get("tenant_id")
    client_id = provider_cfg.get("client_id")
    scopes = provider_cfg.get("scopes") or ["openid", "profile", "email", "User.Read"]
    flow = provider_cfg.get("flow", "device_code")

    if not tenant_id or not client_id:
        raise AuthError("entra.tenant_id / entra.client_id not configured")

    try:
        from msal import PublicClientApplication
    except ImportError as e:
        raise AuthError(
            "msal not installed; add `msal` to dependencies or disable the entra provider"
        ) from e

    # msal injects openid/profile/email automatically; passing them here errors out.
    user_scopes = [s for s in scopes if s.lower() not in ("openid", "profile", "email")]

    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = PublicClientApplication(client_id=client_id, authority=authority)

    loop = asyncio.get_running_loop()

    def _device_code_flow() -> dict:
        df = app.initiate_device_flow(scopes=user_scopes)
        if "user_code" not in df:
            raise AuthError(
                f"could not start device-code flow: {df.get('error_description', df)}"
            )
        msg = df.get("message") or (
            f"To sign in, go to {df.get('verification_uri')} "
            f"and enter code {df.get('user_code')}"
        )
        print(f"\n  {msg}\n", flush=True)
        return app.acquire_token_by_device_flow(df)  # blocks until user completes

    def _interactive_flow() -> dict:
        return app.acquire_token_interactive(scopes=user_scopes)

    result = await loop.run_in_executor(
        None, _interactive_flow if flow == "browser_pkce" else _device_code_flow
    )

    if "access_token" not in result:
        raise AuthError(
            f"entra login failed: {result.get('error_description') or result}"
        )

    claims = result.get("id_token_claims") or {}
    name = claims.get("name") or claims.get("preferred_username") or "user"
    email = (
        claims.get("preferred_username")
        or claims.get("email")
        or claims.get("upn")
    )

    role, role_auth = resolve_role(email, claims, "entra", config)
    caps = resolve_capabilities(role, config)

    return Identity(
        name=str(name),
        email=str(email) if email else None,
        role=role,
        role_authoritativeness=role_auth,
        capabilities=caps,
        provider="entra",
        claims=dict(claims),
    )


# ---------------------------------------------------------------------------
# Google (OAuth 2.0 PKCE via loopback redirect)
# ---------------------------------------------------------------------------

async def login_google(provider_cfg: dict, config: dict) -> Identity:
    """Google login via OAuth 2.0 PKCE on a loopback redirect URI.

    Uses stdlib only (no extra OAuth client). v2 trusts the TLS exchange for
    the id_token; signature verification via Google JWKS is future work.
    """
    import base64
    import hashlib
    import http.server
    import json as _json
    import secrets
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request
    import webbrowser

    client_id = provider_cfg.get("client_id")
    # Optional. NOT needed for Desktop-app-type OAuth clients (Google policy
    # allows them to do PKCE without a secret). REQUIRED for Web-application
    # OAuth clients — Google's token endpoint returns 400 invalid_client when
    # the secret is absent for that client type. Pull from env so it stays
    # out of config.json.
    client_secret = (
        provider_cfg.get("client_secret")
        or os.environ.get("COCO_GOOGLE_CLIENT_SECRET")
    )
    scopes = provider_cfg.get("scopes") or ["openid", "profile", "email"]
    redirect_uri = provider_cfg.get("redirect_uri", "http://localhost:53682/callback")

    if not client_id:
        raise AuthError("google.client_id not configured")

    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    challenge = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(challenge).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)

    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    host = parsed_redirect.hostname or "localhost"
    port = parsed_redirect.port or 53682
    callback_path = parsed_redirect.path or "/callback"

    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urllib.parse.urlencode({
            "client_id": client_id,
            "response_type": "code",
            "scope": " ".join(scopes),
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        })
    )

    received: dict = {}
    done = threading.Event()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != callback_path:
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            received["code"] = (qs.get("code") or [None])[0]
            received["state"] = (qs.get("state") or [None])[0]
            received["error"] = (qs.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Signed in. You can close this tab.</h2></body></html>"
            )
            done.set()

        def log_message(self, *_args, **_kw):  # silence
            return

    server = http.server.HTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        print(
            f"\n  Opening browser for Google sign-in. "
            f"If it doesn't open, visit:\n  {auth_url}\n",
            flush=True,
        )
        try:
            webbrowser.open(auth_url, new=1, autoraise=True)
        except Exception:
            pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: done.wait(timeout=300))
    finally:
        server.shutdown()

    if not done.is_set():
        raise AuthError("google login timed out")
    if received.get("error"):
        raise AuthError(f"google login failed: {received['error']}")
    if received.get("state") != state:
        raise AuthError("google login: state mismatch (possible CSRF)")
    if not received.get("code"):
        raise AuthError("google login: no code received")

    token_params = {
        "code": received["code"],
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": code_verifier,
    }
    if client_secret:
        token_params["client_secret"] = client_secret
    token_body = urllib.parse.urlencode(token_params).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=token_body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    def _post_token() -> dict:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return _json.loads(resp.read().decode())
        except urllib.error.HTTPError as he:
            # Read the response body so Google's actual error message (e.g.
            # invalid_client, redirect_uri_mismatch, invalid_grant) surfaces
            # to the user instead of the bare "HTTP Error 400: Bad Request".
            body = ""
            try:
                body = he.read().decode(errors="replace")
            except Exception:
                pass
            detail = body or he.reason
            hint = ""
            if he.code == 400 and (
                "invalid_client" in body or "client_secret" in body.lower()
            ):
                hint = (
                    " — Web-application OAuth clients require client_secret. "
                    "Either set COCO_GOOGLE_CLIENT_SECRET in .env, or recreate "
                    "the OAuth client as a Desktop app (no secret needed)."
                )
            elif he.code == 400 and "redirect_uri_mismatch" in body:
                hint = (
                    " — the redirect_uri sent in this request does not match "
                    "what's registered in Google Cloud Console for this client."
                )
            raise AuthError(
                f"google token exchange failed (HTTP {he.code}): {detail}{hint}"
            ) from he

    token_resp = await loop.run_in_executor(None, _post_token)

    id_token = token_resp.get("id_token")
    if not id_token:
        raise AuthError("google token response missing id_token")

    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            raise ValueError("malformed JWT")
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = _json.loads(base64.urlsafe_b64decode(payload_b64).decode())
    except Exception as e:
        raise AuthError(f"could not decode id_token: {e}") from e

    name = payload.get("name") or payload.get("given_name") or "user"
    email = payload.get("email")

    role, role_auth = resolve_role(email, payload, "google", config)
    caps = resolve_capabilities(role, config)

    return Identity(
        name=str(name),
        email=str(email) if email else None,
        role=role,
        role_authoritativeness=role_auth,
        capabilities=caps,
        provider="google",
        claims={},
    )
