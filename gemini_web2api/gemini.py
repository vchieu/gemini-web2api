"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import urllib.error
import ssl
import os
import hashlib
import threading

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_httpx_client = None


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


class GeminiError(Exception):
    """Upstream error carrying an optional HTTP status code."""

    def __init__(self, message: str, status: int = None):
        super().__init__(message)
        self.status = status


def fetch_latest_bl():
    """Fetch the latest gemini_bl build label from gemini.google.com."""
    try:
        req = urllib.request.Request(
            "https://gemini.google.com/app",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        ctx = _get_ssl_ctx()
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx),
            )
            resp = opener.open(req, timeout=15)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
        m = re.search(r'(boq_assistant-bard-web-server_\d+\.\d+_p\d+)', html)
        if m:
            return m.group(1)
    except Exception as e:
        log(f"BL auto-update fetch failed: {e}")
    return None


# Refreshing the build label costs a network round-trip. Without coordination,
# every request that hits a 405 would fetch the same page simultaneously.
_bl_lock = threading.Lock()
_bl_last_attempt = 0.0
_BL_RETRY_INTERVAL = 60.0


def update_bl_if_needed() -> bool:
    """Fetch and update gemini_bl when a newer build label is available.

    Never blocks: if another thread is already fetching, or we fetched within
    the last ``_BL_RETRY_INTERVAL`` seconds, this returns False immediately and
    the caller reports its original error instead of queueing behind a fetch.
    """
    global _bl_last_attempt
    if not _bl_lock.acquire(blocking=False):
        return False
    try:
        if time.time() - _bl_last_attempt < _BL_RETRY_INTERVAL:
            return False
        # Stamp before fetching so a failing fetch cools down too.
        _bl_last_attempt = time.time()
        new_bl = fetch_latest_bl()
        if new_bl and new_bl != CONFIG["gemini_bl"]:
            log(f"BL auto-updated: {CONFIG['gemini_bl']} -> {new_bl}")
            CONFIG["gemini_bl"] = new_bl
            return True
        return False
    finally:
        _bl_lock.release()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None and HAS_HTTPX:
        proxy = CONFIG.get("proxy")
        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        _httpx_client = httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True)
    return _httpx_client


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "", None
    try:
        mtime = os.path.getmtime(cookie_file)
        if mtime == _cookie_cache["mtime"] and _cookie_cache["str"]:
            return _cookie_cache["str"], _cookie_cache["sapisid"]
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return _cookie_cache["str"], _cookie_cache["sapisid"]


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


def _build_headers() -> dict:
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    return headers


def _apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        # Match Gemini Web temporary-chat requests.
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    return urllib.parse.urlencode(params)


def _get_url() -> str:
    reqid = int(time.time()) % 1000000
    account_prefix = _account_prefix()
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


_SCAFFOLD_FENCE = re.compile(r"```\w*\?code_")
_SCAFFOLD_LANGS = ("python", "javascript", "text")


def _holds_scaffold(buf: str) -> bool:
    """True while ``buf`` ends inside an open *upstream* scaffolding fence.

    Upstream interleaves helper blocks such as
    ```` ```python?code_reference&code_event_index=0 ... ``` ```` that
    ``clean_text`` strips as a unit, so those must stay buffered until they
    close. Ordinary code fences in the answer must not be buffered -- counting
    backticks instead would hold an entire legitimate code block back until its
    closing fence arrived, delivering it as one late lump.
    """
    if buf.count("```") % 2 == 0:
        return False
    open_at = buf.rfind("```")
    rest = buf[open_at + 3:]
    head, terminated, _ = rest.partition("\n")
    if _SCAFFOLD_FENCE.match(buf[open_at:]):
        return True                      # ```python?code_... arrived complete
    if terminated:
        return False                     # tag finished without ?code_ → normal fence
    if "?" in head:
        return True                      # tag still growing past '?'
    # Language tag not terminated yet: hold only while it can still grow into
    # one of the scaffolding tags (released as soon as a newline proves the
    # fence is an ordinary one).
    return any(lang.startswith(head) for lang in _SCAFFOLD_LANGS)


def _extract_texts_from_line(line: str) -> list:
    """Parse a single wrb.fr line and return list of text strings found."""
    if '"wrb.fr"' not in line or len(line) < 200:
        return []
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str or len(inner_str) < 50:
            return []
        inner = json.loads(inner_str)
        if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
            return []
        texts = []
        for part in inner[4]:
            if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                for t in part[1]:
                    if isinstance(t, str) and t:
                        texts.append(t)
        return texts
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text."""
    bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', raw)
    if bard_err:
        raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]")
    last_text = ""
    for line in raw.split("\n"):
        for t in _extract_texts_from_line(line):
            if len(t) > len(last_text):
                last_text = t
    return clean_text(last_text)


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    """Non-streaming generation with BL-aware retry."""
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    bl_refreshed = False
    for attempt in range(CONFIG["retry_attempts"]):
        url = _get_url()
        headers = _build_headers()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            raw = resp.read().decode("utf-8", errors="replace")
            return extract_response_text(raw)
        except urllib.error.HTTPError as e:
            last_err = GeminiError(f"HTTP {e.code} from Gemini upstream", status=e.code)
            # A stale BL build label manifests as 405/404: refresh and retry once.
            if e.code in (404, 405) and not bl_refreshed and update_bl_if_needed():
                bl_refreshed = True
                log("BL updated after upstream error, retrying")
                continue
            # Client errors will not succeed on retry.
            if e.code in (400, 401, 403, 404, 405):
                break
        except Exception as e:
            last_err = GeminiError(str(e))
        if attempt < CONFIG["retry_attempts"] - 1:
            log(f"Retry {attempt + 1}/{CONFIG['retry_attempts']}: {last_err}")
            time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None):
    """Streaming generation via httpx with BL-aware retry.

    Text is buffered only while an *upstream scaffolding* fence is open so that
    blocks (````python?code_...````) are always stripped whole, even when they
    straddle two network chunks. Ordinary code fences stream through
    immediately.
    """
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        if text:
            yield text
        return

    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
    client = _get_httpx_client()

    last_err = None
    bl_refreshed = False
    for attempt in range(CONFIG["retry_attempts"]):
        url = _get_url()
        headers = _build_headers()
        emitted_raw_text = ""
        clean_buf = ""
        emitted_any = False
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                        if bard_err:
                            raise RuntimeError(
                                f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]"
                            )
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        texts = _extract_texts_from_line(line)
                        for index, t in enumerate(texts):
                            if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                continue
                            if index:
                                # Later text in the *same* line is a separate
                                # block (thinking summary), not a cumulative
                                # echo of the earlier one: emit it immediately.
                                # Keep the primary cumulative text tracking
                                # intact so the next line still has a prefix
                                # relation and does not log spurious warnings.
                                block = clean_text(t, strip=False)
                                if block:
                                    emitted_any = True
                                    yield block
                                continue
                            if t.startswith(emitted_raw_text):
                                clean_buf += t[len(emitted_raw_text):]
                                emitted_raw_text = t
                            else:
                                # First text of the line disagrees with everything
                                # emitted so far, i.e. upstream restarted. Keep the
                                # stream alive rather than failing the request; the
                                # sample lets a real payload be inspected later.
                                log(f"Stream segment without prefix relation, "
                                    f"skipping: {t[:120]!r}")
                                continue
                            if _holds_scaffold(clean_buf):
                                # Inside an open upstream scaffolding block:
                                # hold until it closes so clean_text can drop it
                                # in one piece.
                                continue
                            delta = clean_text(clean_buf, strip=False)
                            clean_buf = ""
                            if delta:
                                emitted_any = True
                                yield delta
            if clean_buf:
                # If the stream ended inside an open scaffolding fence, strip it
                if _holds_scaffold(clean_buf):
                    open_at = clean_buf.rfind("```")
                    clean_buf = clean_buf[:open_at]
                delta = clean_text(clean_buf, strip=False)
                if delta:
                    yield delta
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            last_err = GeminiError(f"HTTP {status} from Gemini upstream", status=status)
            if status in (404, 405) and not bl_refreshed and update_bl_if_needed():
                bl_refreshed = True
                log("BL updated after upstream error, retrying stream")
                continue
            if status in (400, 401, 403, 404, 405):
                raise last_err
        except Exception as e:
            last_err = GeminiError(str(e))
            # Never retry once partial content has been sent to the client.
            if emitted_any:
                raise last_err
        if attempt < CONFIG["retry_attempts"] - 1:
            log(f"Stream retry {attempt + 1}/{CONFIG['retry_attempts']}: {last_err}")
            time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
