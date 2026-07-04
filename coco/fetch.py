"""URL ingestion skill — fetch + extract + markdown + image processing.

Pipeline (called from `agent._handle_ingest`):
  1. httpx GET (size + content-type checked, redirects followed)
  2. readability-lxml extracts the main article HTML
  3. <img> tags walked → image manifest of placeholders + asset URLs
  4. markdownify converts the article to markdown (placeholders survive as text)
  5. asyncio.gather over image URLs: fetch, downscale via Pillow, base64-encode

Returns a `FetchResult` with markdown + dict[placeholder → ImageBlob]. Image blobs
carry a ready-to-substitute `data:` URI. Failures are raised as `FetchError`; per-image
failures are silently dropped from the manifest (logged on the tracing span).
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from lxml import html as lxml_html
from markdownify import markdownify as md
from PIL import Image
from readability import Document

from . import tracing


class FetchError(Exception):
    """Fatal fetch failures — bad URL, non-HTML, oversize page, network."""


@dataclass
class ImageBlob:
    src: str
    alt: str | None
    mime: str
    data_b64: str
    original_bytes: int
    post_downscale_bytes: int
    dimensions: tuple[int, int]

    @property
    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.data_b64}"


@dataclass
class FetchResult:
    url: str
    title: str | None
    markdown: str
    images: dict[str, ImageBlob] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    truncated: bool = False


_ALLOWED_HTML_TYPES = ("text/html", "application/xhtml+xml")
_PLACEHOLDER_RE = re.compile(r"\[IMG_(\d+)\]")


async def fetch_url(url: str, config: dict) -> FetchResult:
    """Top-level entry — fetch, extract, manifest, encode."""
    with tracing.observation(
        "url_ingest_fetch",
        as_type="span",
        input={"url": url},
    ) as span:
        html_bytes, final_url, _content_type = await _get_html(url, config)
        article_html, title = _extract_article(html_bytes, final_url, config)
        article_html_with_placeholders, raw_manifest = _walk_and_placeholder(
            article_html, final_url, config
        )
        markdown = _to_markdown(article_html_with_placeholders)

        truncated = False
        max_chars = config["ingest_markdown_max_chars"]
        if len(markdown) > max_chars:
            markdown = _truncate_at_boundary(markdown, max_chars)
            truncated = True

        images = await _process_images(raw_manifest, config)
        # Drop placeholders that didn't survive image processing — the LLM
        # should never see a placeholder we can't substitute.
        markdown = _drop_dead_placeholders(markdown, images)

        tracing.update(span, output={
            "final_url": final_url,
            "title": title,
            "markdown_chars": len(markdown),
            "truncated": truncated,
            "candidates": len(raw_manifest),
            "images_kept": len(images),
            "kept_ids": sorted(images.keys()),
        })
        return FetchResult(
            url=final_url,
            title=title,
            markdown=markdown,
            images=images,
            errors=[],
            truncated=truncated,
        )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


async def _get_html(url: str, config: dict) -> tuple[bytes, str, str]:
    headers = {
        "User-Agent": config["ingest_user_agent"],
        "Accept": "text/html,application/xhtml+xml",
    }
    timeout = config["ingest_request_timeout_s"]
    max_bytes = config["ingest_max_page_bytes"]

    with tracing.observation("fetch_html", as_type="span", input={"url": url}) as span:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=timeout, headers=headers
            ) as client:
                resp = await client.get(url)
        except httpx.TimeoutException as e:
            raise FetchError("timeout") from e
        except httpx.RequestError as e:
            raise FetchError(f"network: {e}") from e

        if resp.status_code >= 400:
            raise FetchError(f"status={resp.status_code}")

        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type and not any(content_type.startswith(t) for t in _ALLOWED_HTML_TYPES):
            raise FetchError(f"not_html: {content_type or '(missing)'}")

        body = resp.content
        if len(body) > max_bytes:
            raise FetchError(f"page_too_large: {len(body)} bytes")

        final_url = str(resp.url)
        tracing.update(span, output={
            "status": resp.status_code,
            "bytes": len(body),
            "content_type": content_type,
            "final_url": final_url,
        })
        return body, final_url, content_type


def _extract_article(html_bytes: bytes, base_url: str, config: dict) -> tuple[str, str | None]:
    with tracing.observation(
        "extract_article", as_type="span", input={"bytes": len(html_bytes)}
    ) as span:
        # readability-lxml expects a str, not bytes; decode with a tolerant fallback.
        if isinstance(html_bytes, bytes):
            try:
                html_text = html_bytes.decode("utf-8")
            except UnicodeDecodeError:
                html_text = html_bytes.decode("utf-8", errors="replace")
        else:
            html_text = html_bytes
        try:
            doc = Document(html_text)
            title = doc.short_title()
            article_html = doc.summary(html_partial=True)
        except Exception as e:
            raise FetchError(f"readability_failed: {e}") from e

        text_chars = len(re.sub(r"<[^>]+>", "", article_html).strip())
        tracing.update(span, output={
            "title": title,
            "article_html_chars": len(article_html),
            "article_text_chars": text_chars,
        })
        if text_chars < config["ingest_min_article_chars"]:
            raise FetchError("no_readable_content")
        return article_html, title


def _walk_and_placeholder(
    article_html: str, base_url: str, config: dict
) -> tuple[str, dict[str, dict]]:
    """Rewrite <img> tags to [IMG_n] placeholders; return (modified_html, manifest)."""
    max_imgs = config["ingest_max_images_per_page"]
    try:
        tree = lxml_html.fragment_fromstring(article_html, create_parent="div")
    except Exception:
        return article_html, {}

    manifest: dict[str, dict] = {}
    counter = 0

    for img in list(tree.iter("img")):
        if counter >= max_imgs:
            _drop_element(img)
            continue
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            _drop_element(img)
            continue
        abs_src = urljoin(base_url, src)
        parsed = urlparse(abs_src)
        if not parsed.scheme.startswith("http"):
            _drop_element(img)
            continue

        counter += 1
        key = f"IMG_{counter}"
        manifest[key] = {"src": abs_src, "alt": img.get("alt") or None}

        placeholder = f" [{key}] "
        parent = img.getparent()
        if parent is None:
            _drop_element(img)
            continue
        # Splice the placeholder into the surrounding text.
        idx = list(parent).index(img)
        if idx == 0:
            parent.text = (parent.text or "") + placeholder
        else:
            prev = parent[idx - 1]
            prev.tail = (prev.tail or "") + placeholder
        parent.remove(img)

    new_html = lxml_html.tostring(tree, encoding="unicode")
    return new_html, manifest


def _drop_element(el) -> None:
    parent = el.getparent()
    if parent is not None:
        parent.remove(el)


def _to_markdown(html: str) -> str:
    return md(html, heading_style="ATX")


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < max_chars * 0.5:
        cut = max_chars
    return text[:cut].rstrip() + "\n\n[…truncated…]\n"


async def _process_images(
    manifest: dict[str, dict], config: dict
) -> dict[str, ImageBlob]:
    if not manifest:
        return {}

    with tracing.observation(
        "process_images", as_type="span", input={"candidates": len(manifest)}
    ) as span:
        sem = asyncio.Semaphore(config["ingest_image_concurrency"])
        timeout = config["ingest_request_timeout_s"]
        headers = {"User-Agent": config["ingest_user_agent"]}

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, headers=headers
        ) as client:
            async def fetch_one(key: str, entry: dict) -> tuple[str, ImageBlob | None, str | None]:
                async with sem:
                    try:
                        resp = await client.get(entry["src"])
                    except httpx.TimeoutException:
                        return key, None, "timeout"
                    except httpx.RequestError as e:
                        return key, None, f"network: {e}"
                if resp.status_code >= 400:
                    return key, None, f"status={resp.status_code}"

                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                allowed = config["ingest_allowed_image_mimes"]
                if ctype not in allowed:
                    guess = _guess_mime_from_url(entry["src"])
                    if guess in allowed:
                        ctype = guess
                    else:
                        return key, None, f"mime_disallowed: {ctype or '(missing)'}"

                blob = _maybe_downscale(resp.content, ctype, config)
                if blob is None:
                    return key, None, "downscale_failed_or_oversize"

                b64 = base64.b64encode(blob["bytes"]).decode("ascii")
                return key, ImageBlob(
                    src=entry["src"],
                    alt=entry.get("alt"),
                    mime=blob["mime"],
                    data_b64=b64,
                    original_bytes=len(resp.content),
                    post_downscale_bytes=len(blob["bytes"]),
                    dimensions=blob["dimensions"],
                ), None

            results = await asyncio.gather(
                *[fetch_one(k, v) for k, v in manifest.items()],
                return_exceptions=True,
            )

        kept: dict[str, ImageBlob] = {}
        dropped: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                dropped.append({"key": "?", "reason": f"task_exception: {r}"})
                continue
            key, blob, reason = r
            if blob is None:
                dropped.append({"key": key, "reason": reason or "?"})
            else:
                kept[key] = blob

        tracing.update(span, output={
            "kept": len(kept),
            "dropped": len(dropped),
            "dropped_reasons": dropped[:20],
        })
        return kept


def _guess_mime_from_url(url: str) -> str | None:
    ext = url.rsplit(".", 1)[-1].lower().split("?")[0]
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }.get(ext)


def _maybe_downscale(raw: bytes, mime: str, config: dict) -> dict | None:
    """Decode → downscale (if oversized) → re-encode → size-check.

    Returns dict {bytes, mime, dimensions} or None on failure / over-cap.
    """
    max_dim = config["ingest_image_max_dim"]
    max_bytes = config["ingest_image_max_bytes"]

    if mime == "image/svg+xml":
        if len(raw) > max_bytes:
            return None
        return {"bytes": raw, "mime": mime, "dimensions": (0, 0)}

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return None

    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / longest
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    fmt_map = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }
    fmt = fmt_map.get(mime)
    if fmt is None:
        return None

    save_kwargs: dict = {}
    if fmt == "JPEG":
        if img.mode != "RGB":
            img = img.convert("RGB")
        save_kwargs["quality"] = 85
        save_kwargs["optimize"] = True
    elif fmt == "PNG":
        if img.mode == "P":
            img = img.convert("RGBA")
        save_kwargs["optimize"] = True
    elif fmt == "WEBP":
        save_kwargs["quality"] = 85
        save_kwargs["method"] = 4

    out = io.BytesIO()
    try:
        img.save(out, format=fmt, **save_kwargs)
    except Exception:
        return None
    data = out.getvalue()

    if len(data) > max_bytes and fmt in {"JPEG", "WEBP"}:
        out = io.BytesIO()
        save_kwargs["quality"] = 60
        try:
            img.save(out, format=fmt, **save_kwargs)
            data = out.getvalue()
        except Exception:
            return None

    if len(data) > max_bytes:
        return None
    return {"bytes": data, "mime": mime, "dimensions": (w, h)}


def _drop_dead_placeholders(markdown: str, kept: dict[str, ImageBlob]) -> str:
    def repl(m: re.Match) -> str:
        key = f"IMG_{m.group(1)}"
        return m.group(0) if key in kept else ""
    return _PLACEHOLDER_RE.sub(repl, markdown)
