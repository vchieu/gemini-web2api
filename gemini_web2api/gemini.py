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
from .tools import looks_like_tool_call

_ssl_ctx = None
# A cache entry is a single immutable tuple, never a dict of separate keys:
# readers grab one reference (atomic under the GIL), so `str` from one file
# generation can never be paired with `sapisid`/`mtime` from another while a
# second thread is mid-update.
_cookie_cache = ("", None, 0)
_httpx_client = None
# Client construction is not idempotent. The server is threaded, so two
# requests racing on the `None` check would each build an httpx.Client and one
# of them would be dropped without ever being closed.
_httpx_lock = threading.Lock()


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        # A file sink keeps a trail when stderr is swallowed by a launcher
        # (Start-Process, nohup, systemd without journal capture).
        path = CONFIG.get("log_file")
        if path:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass


class GeminiError(Exception):
    """Upstream error carrying an optional HTTP status code."""

    def __init__(self, message: str, status: int = None):
        super().__init__(message)
        self.status = status


def _dump_raw(raw: str):
    """Write one unfiltered upstream response to ``debug_raw_file``.

    ``generate()`` only ever hands back the extracted text, so the shape of the
    raw payload (how many elements ``inner[4]`` holds, which one is thinking vs
    answer vs tool call) is invisible from the logs. Enabled by
    ``CONFIG["debug_raw"]``; failures are silent because this is diagnostics.
    """
    path = CONFIG.get("debug_raw_file") or "gemini-raw.log"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== [{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"{len(raw)} bytes =====\n")
            fh.write(raw)
            fh.write("\n")
    except OSError:
        pass


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
        # The label family is not stable: Google renamed
        # ``boq_assistant-bard-web-server_20260716.08_p0`` to
        # ``boq_gemini-web-uiserver_20261007.12_p0`` when the UI was rebuilt,
        # and a family-specific pattern then matches nothing -- auto-update
        # silently stops working and every request uses the stale default.
        # Match the shape (``boq_<family>_<date>.<build>_p<n>``) instead.
        m = re.search(r'(boq_[a-z0-9-]+_\d+\.\d+_p\d+)', html)
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
        with _httpx_lock:
            # Re-checked inside the lock: only the thread that created the
            # client assigns it, the rest reuse the same one.
            if _httpx_client is None:
                proxy = CONFIG.get("proxy")
                transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
                _httpx_client = httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True)
    return _httpx_client


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    global _cookie_cache
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "", None
    cached_str, cached_sapisid, cached_mtime = _cookie_cache
    try:
        mtime = os.path.getmtime(cookie_file)
        if mtime == cached_mtime and cached_str:
            return cached_str, cached_sapisid
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
            # The bundled extension exports {cookie, sapisid, auth_user,
            # xsrf_token, gemini_bl} in one file. Ignore the extra fields and
            # StreamGenerate goes out without its `at` form field (HTTP 400)
            # and with whatever stale build label happens to be configured,
            # so sync them here: pointing cookie_file at the export must be
            # enough on its own.
            if data.get("xsrf_token"):
                CONFIG["xsrf_token"] = data["xsrf_token"]
            if data.get("auth_user") not in (None, ""):
                CONFIG["auth_user"] = data["auth_user"]
            if data.get("gemini_bl"):
                CONFIG["gemini_bl"] = data["gemini_bl"]
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        # One atomic swap: a reader either sees the whole old entry or the
        # whole new one.
        _cookie_cache = (cookie_str, sapisid or None, mtime)
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return cached_str, cached_sapisid


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


def _build_headers(ticket: str = None) -> dict:
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
    if ticket:
        # The per-family ticket is what the upstream actually routes on; the
        # f.req [79]/[80] fields are advisory without it.
        from .models import TICKET_HEADER
        headers[TICKET_HEADER] = ticket
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
    # The `at` (xsrf) token lives in the cookie file and only reaches CONFIG
    # via load_cookie(). The body is built before any header is assembled, so
    # without this call the first attempt of a request goes out without `at`
    # and Gemini answers HTTP 400 -- and 400 is treated as non-retryable.
    load_cookie()
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
    # One line per request naming what was actually asked for: when the answer
    # looks like another model, this plus check_routing is the whole trail.
    log(f"Upstream model family={model_id} variant={(extra_fields or {}).get(80)}")
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    if CONFIG.get("debug_raw"):
        # Mirror the response dump: the built prompt (with tool block and any
        # injected extra fields) is what the model actually sees, so a collapse
        # between client request and upstream text is visible right here.
        _dump_raw("REQUEST >>>\n" + json.dumps(inner, ensure_ascii=False)
                  + "\n<<< REQUEST")
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


# Two payload shapes carry the same upstream rejection: the legacy prose form
# ``BardErrorInfo [1099]`` and the protobuf-JSON form the current build emits
# (``...application.BardErrorInfo",[1099]]``). The second one does not match
# ``BardErrorInfo\s*\[`` -- a quote and a comma sit between the name and the
# bracket -- so the rejection used to fall through as an empty reply and the
# real reason (often transient: the same request succeeds minutes later)
# never reached the log or the client.
_BARD_ERROR_RE = re.compile(r'BardErrorInfo["\]]*[\s,]*\[(\d+)\]')

# Codes Google is known to return, appended as a human-readable hint.
BARD_ERROR_HINTS = {
    1013: "temporary generation error",
    1037: "usage limit exceeded for the requested model",
    1050: "requested model is inconsistent with the conversation",
    1052: "requested model header is invalid or unavailable",
    1060: "Google temporarily blocked this IP address",
}


def bard_error_code(raw: str):
    """Return the BardErrorInfo code embedded in ``raw``, or None."""
    m = _BARD_ERROR_RE.search(raw)
    return int(m.group(1)) if m else None


def raise_bard_error(raw: str) -> None:
    """Raise a GeminiError when ``raw`` carries a BardErrorInfo rejection.

    Returns silently otherwise. GeminiError (status None) maps to 503 on the
    OpenAI endpoints: the upstream -- not the client's request -- refused.
    """
    code = bard_error_code(raw)
    if code is None:
        return
    hint = BARD_ERROR_HINTS.get(code)
    raise GeminiError(
        f"Gemini upstream rejected request: BardErrorInfo [{code}]"
        + (f" -- {hint}" if hint else "")
    )


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text.

    Ties break towards the candidate carrying a tool call, not towards the
    longer one. The answer and the call arrive as separate texts within the
    same payload; if a thinking or summary block happens to run longer, a
    longest-wins pick drops the call and the client is handed prose that only
    claims a tool was used. Nothing changes when no candidate looks like a
    call, which is the overwhelmingly common case.
    """
    raise_bard_error(raw)
    best_text = ""
    best_call = ""
    for line in raw.split("\n"):
        for t in _extract_texts_from_line(line):
            if looks_like_tool_call(t):
                if len(t) > len(best_call):
                    best_call = t
            elif len(t) > len(best_text):
                best_text = t
    return clean_text(best_call or best_text)


def upstream_echo(raw: str):
    """Return the (label, family, variant) the upstream echoed, or None.

    The StreamGenerate response names the model that actually served the
    request; comparing it against what was requested is the only way to see a
    silent misroute (the request still answers 200 with fluent text).
    """
    for line in raw.split("\n"):
        if '"wrb.fr"' not in line or len(line) < 200:
            continue
        try:
            meta = json.loads(json.loads(line)[0][2])
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
        if isinstance(meta, list) and len(meta) >= 60:
            return meta[42], meta[58], meta[59]
    return None


def check_routing(raw: str, model_id: int, extra_fields: dict = None, ticket: str = None) -> None:
    """Log a warning when upstream served a different model than requested.

    A missing or expired model ticket is what causes this: the upstream then
    ignores [79]/[80] and answers with the account default (Pro). The answer
    itself is fine, so nothing else in the pipeline notices -- hence a log
    line rather than an error.
    """
    echo = upstream_echo(raw)
    if not echo:
        return
    _, fam, var = echo
    if ticket:
        # The ticket wins over the body fields, so expectations are read from
        # the (family, variant) embedded in the ticket actually sent.
        try:
            t = json.loads(ticket)
            want_fam, want_var = t[14], t[15]
        except (json.JSONDecodeError, IndexError, TypeError):
            want_fam, want_var = model_id, (extra_fields or {}).get(80)
    else:
        want_fam, want_var = model_id, (extra_fields or {}).get(80)
    if fam != want_fam or (want_var is not None and var != want_var):
        log(f"Routing mismatch: requested family={want_fam} variant={want_var} "
            f"but upstream served {echo[0]!r} (family={fam} variant={var}); "
            f"the model ticket in CONFIG['model_tickets'] may be missing or "
            f"expired -- refresh it from a fresh browser capture")


def _status_error(status) -> GeminiError:
    """Build the GeminiError for an upstream HTTP status.

    A 400 with no cookie configured is rarely the client's fault: Google
    rejects cookie-less (anonymous) sessions with 400, and reporting that
    verbatim tells the caller their request was malformed when the real fix
    is on this side.
    """
    msg = f"HTTP {status} from Gemini upstream"
    if status == 400 and not CONFIG.get("cookie_file"):
        msg += ("; no cookie_file is set and Google currently rejects"
                " anonymous sessions -- export cookies via the bundled"
                " extension and set cookie_file")
    return GeminiError(msg, status=status)


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, ticket: str = None) -> str:
    """Non-streaming generation with BL-aware retry."""
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    bl_refreshed = False
    for attempt in range(CONFIG["retry_attempts"]):
        url = _get_url()
        headers = _build_headers(ticket)
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
            if CONFIG.get("debug_raw"):
                _dump_raw(raw)
            check_routing(raw, model_id, extra_fields, ticket)
            return extract_response_text(raw)
        except urllib.error.HTTPError as e:
            last_err = _status_error(e.code)
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


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, ticket: str = None):
    """Streaming generation via httpx with BL-aware retry.

    Text is buffered only while an *upstream scaffolding* fence is open so that
    blocks (````python?code_...````) are always stripped whole, even when they
    straddle two network chunks. Ordinary code fences stream through
    immediately.
    """
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields, ticket)
        if text:
            yield text
        return

    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
    client = _get_httpx_client()

    last_err = None
    bl_refreshed = False
    for attempt in range(CONFIG["retry_attempts"]):
        url = _get_url()
        headers = _build_headers(ticket)
        emitted_raw_text = ""
        clean_buf = ""
        emitted_any = False
        raw_full = ""
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    raw_full += chunk
                    if "BardErrorInfo" in buf:
                        raise_bard_error(buf)
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
            if CONFIG.get("debug_raw") and raw_full:
                # Mirror generate(): the stream path also dumps the raw
                # upstream frames so prefix/delta handling can be audited.
                _dump_raw(raw_full)
            check_routing(raw_full, model_id, extra_fields, ticket)
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            last_err = _status_error(status)
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
    if CONFIG.get("debug_raw") and raw_full:
        _dump_raw(raw_full)
    raise last_err
