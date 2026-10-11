"""Multimodal: Scotty resumable upload for Gemini image input."""
import json
import base64
import ipaddress
import socket
import urllib.request
import urllib.parse
import time
import ssl
import re
from urllib.parse import urlparse

from .config import CONFIG, _config_lock_ctx
from .gemini import load_cookie, make_sapisidhash, _get_ssl_ctx, log


def _get_page_tokens() -> dict:
    """Fetch WIZ_global_data tokens from Gemini page (Push-ID, X-Client-Pctx)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    try:
        req = urllib.request.Request("https://gemini.google.com/app", headers=headers)
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
            )
            resp = opener.open(req, timeout=30)
        else:
            resp = urllib.request.urlopen(req, context=_get_ssl_ctx(), timeout=30)
        html = resp.read().decode()
        tokens = {}
        for key, pattern in [
            ("push_id", r'"qKIAYe":"([^"]+)"'),
            ("pctx", r'"Ylro7b":"([^"]+)"'),
            ("at", r'"thykhd":"([^"]+)"'),
        ]:
            m = re.search(pattern, html)
            if m:
                tokens[key] = m.group(1)
        # The xsrf token ages out like everything else on the page: refresh
        # CONFIG from the same /app fetch the image path already does, so a
        # stale `at` form field stops failing StreamGenerate with HTTP 400.
        m = re.search(r'"SNlM0e":"([^"]+)"', html)
        if m:
            with _config_lock_ctx():
                CONFIG["xsrf_token"] = m.group(1)
            log("xsrf_token refreshed from the Gemini page")
        return tokens
    except Exception as e:
        log(f"Page token fetch failed: {e}")
        return {}


_page_tokens_cache = {"tokens": {}, "ts": 0}


def _cached_page_tokens() -> dict:
    now = time.time()
    if now - _page_tokens_cache["ts"] > 600:
        _page_tokens_cache["tokens"] = _get_page_tokens()
        _page_tokens_cache["ts"] = now
    return _page_tokens_cache["tokens"]


def detect_image_mime(image_bytes: bytes, fallback: str = "image/png") -> str:
    """Infer a common raster image MIME type from its file signature."""
    if not isinstance(image_bytes, bytes):
        return fallback
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(image_bytes) >= 12 and image_bytes[4:8] == b"ftyp":
        brand = image_bytes[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"hevc", b"hevx"):
            return "image/heic"
    return fallback


def upload_image(image_bytes: bytes, filename: str = "image.png", mime_type: str = "image/png") -> str:
    """Upload image via Scotty resumable upload. Returns file reference path."""
    tokens = _cached_page_tokens()
    push_id = tokens.get("push_id", "feeds/mcudyrk2a4khkz")
    pctx = tokens.get("pctx", "CgcSBWjK7pYx")

    cookie_str, sapisid = load_cookie()
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    # Step 1: Initiate resumable upload
    start_headers = {
        "Push-ID": push_id,
        "X-Tenant-Id": "bard-storage",
        "X-Client-Pctx": pctx,
        "X-Goog-Upload-Header-Content-Length": str(len(image_bytes)),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if cookie_str:
        start_headers["Cookie"] = cookie_str
    if sapisid:
        start_headers["Authorization"] = make_sapisidhash(sapisid)

    start_url = "https://content-push.googleapis.com/upload/"
    req = urllib.request.Request(start_url, data=b"", headers=start_headers, method="POST")

    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=ctx)
        )
        resp = opener.open(req, timeout=30)
    else:
        resp = urllib.request.urlopen(req, context=ctx, timeout=30)

    upload_url = resp.headers.get("X-Goog-Upload-URL") or resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError(f"No upload URL in response headers: {dict(resp.headers)}")

    log(f"Upload session started: {upload_url[:80]}...")

    # Step 2: Upload file data + finalize
    upload_headers = {
        "X-Goog-Upload-Command": "upload, finalize",
        "X-Goog-Upload-Offset": "0",
        "Content-Type": "application/octet-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    req2 = urllib.request.Request(upload_url, data=image_bytes, headers=upload_headers, method="POST")
    if proxy:
        resp2 = opener.open(req2, timeout=60)
    else:
        resp2 = urllib.request.urlopen(req2, context=ctx, timeout=60)

    file_ref = resp2.read().decode().strip()
    if not file_ref or not file_ref.startswith("/"):
        raise RuntimeError(f"Invalid file reference: {file_ref[:100]}")

    log(f"Image uploaded: {filename} -> {file_ref[:50]}...")
    return file_ref


# Upper bound for a fetched image body. Without it one URL can eat unbounded
# RAM while the server reads it; the caller turns b"" into "image fetch
# failed", so oversize is a failed fetch, not a crash.
_MAX_IMAGE_BYTES = 16 * 1024 * 1024


def _resolve_public_ip(hostname: str) -> str | None:
    """Resolve hostname and return a single public IP, or None if blocked.

    This resolves DNS *once* and validates all addresses are global.
    The caller must then connect directly to this IP to prevent
    DNS-rebinding / TOCTOU attacks.
    """
    if not hostname:
        return None
    host = hostname.strip().rstrip(".").strip("[]")
    # Literal IP: check directly.
    try:
        return host if ipaddress.ip_address(host).is_global else None
    except ValueError:
        pass
    # Hostname: resolve and validate all addresses.
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return None
    if not infos:
        return None
    public_ip = None
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
            if not addr.is_global:
                return None  # Any non-global address blocks the whole host.
            public_ip = str(addr)  # Remember first global address.
        except ValueError:
            return None
    return public_ip


def _host_is_private(hostname: str) -> bool:
    """True when ``hostname`` resolves only to non-public IP addresses.

    Delegates to ``_resolve_public_ip`` for the actual check.
    """
    return _resolve_public_ip(hostname) is None


def fetch_image_bytes(url: str) -> bytes:
    """Fetch image from URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        log(f"Image fetch skipped for unsupported URL scheme: {parsed.scheme or 'none'}")
        return b""
    if _host_is_private(parsed.hostname or ""):
        log(f"Image fetch blocked for non-public target: {parsed.hostname or 'none'}")
        return b""
    # Resolve IP once and connect directly to prevent DNS-rebinding / TOCTOU.
    target_ip = _resolve_public_ip(parsed.hostname or "")
    if not target_ip:
        log(f"Image fetch blocked: could not resolve public IP for {parsed.hostname}")
        return b""
    try:
        # Reconstruct URL with resolved IP, keeping original Host header.
        # This avoids fragile string replacement on the full URL.
        netloc = f"{target_ip}:{parsed.port}" if parsed.port else target_ip
        reconstructed = parsed._replace(netloc=netloc).geturl()
        req = urllib.request.Request(
            reconstructed,
            headers={"User-Agent": "Mozilla/5.0", "Host": parsed.hostname},
            method="GET",
        )
        proxy = CONFIG.get("proxy")
        if proxy:
            # With proxy, we can't pin the IP; rely on proxy's DNS.
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
            )
            with opener.open(req, timeout=30) as resp:
                data = resp.read(_MAX_IMAGE_BYTES + 1)
        else:
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
            )
            with opener.open(req, timeout=30) as resp:
                data = resp.read(_MAX_IMAGE_BYTES + 1)
        if len(data) > _MAX_IMAGE_BYTES:
            log(f"Image fetch failed: body exceeds {_MAX_IMAGE_BYTES} bytes")
            return b""
        return data
    except Exception as e:
        log(f"Image fetch failed: {e}")
        return b""
