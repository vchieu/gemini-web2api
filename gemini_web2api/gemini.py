"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import ssl
import os
import hashlib

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
            # Cookie file may also carry auth_user / xsrf_token / gemini_bl
            # (same JSON shape as in the issue). Sync them into CONFIG,
            # otherwise StreamGenerate requests go out without the `at`
            # param and with a stale `bl`, and Gemini answers 400.
            if data.get("xsrf_token"):
                CONFIG["xsrf_token"] = data["xsrf_token"]
            if "auth_user" in data and data["auth_user"] not in (None, ""):
                CONFIG["auth_user"] = data["auth_user"]
            if data.get("gemini_bl"):
                CONFIG["gemini_bl"] = data["gemini_bl"]
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
    log(f"Upstream model family={model_id} variant={(extra_fields or {}).get(80)}")
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


def _extract_auth_from_html(html: str) -> tuple:
    """Extract (xsrf_token, gemini_bl) from Gemini app page HTML."""
    xsrf = None
    m = re.search(r'"SNlM0e"\s*:\s*"([^"]+)"', html)
    if m:
        raw = m.group(1)
        try:
            xsrf = raw.encode().decode("unicode_escape")
        except Exception:
            xsrf = raw
        xsrf = xsrf.replace("\\u003d", "=").replace("\\u0026", "&")
    bl = None
    b = re.search(r"(boq_assistant-bard-web-server_\d+\.\d+_p\d+)", html)
    if b:
        bl = b.group(1)
    return xsrf, bl


def _persist_auth_to_file(xsrf: str | None, bl: str | None) -> None:
    """Write refreshed xsrf_token/gemini_bl back to the JSON cookie file."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return
    try:
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if not content.startswith("{"):
            return
        data = json.loads(content)
        if xsrf:
            data["xsrf_token"] = xsrf
        if bl:
            data["gemini_bl"] = bl
        with open(cookie_file, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        _cookie_cache["mtime"] = 0
    except Exception as e:
        log(f"Auth persist failed: {e}")


def refresh_auth() -> bool:
    """Refresh xsrf_token/gemini_bl from the authenticated Gemini page.

    SNlM0e rotates every few minutes, so a statically exported token goes
    stale and Gemini answers 400 with an xsrf error. Re-fetch it with the
    cookie session. Returns True if a usable token was obtained.
    """
    cookie_str, sapisid = load_cookie()
    if not cookie_str:
        return False
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cookie": cookie_str,
        }
        if sapisid:
            headers["Authorization"] = make_sapisidhash(sapisid)
        url = f"https://gemini.google.com{_account_prefix()}/app"
        ctx = _get_ssl_ctx()
        proxy = CONFIG.get("proxy")
        req = urllib.request.Request(url, headers=headers, method="GET")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx)
            )
            resp = opener.open(req, timeout=30)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=30)
        html = resp.read().decode("utf-8", errors="replace")
        xsrf, bl = _extract_auth_from_html(html)
        if xsrf and xsrf != CONFIG.get("xsrf_token"):
            CONFIG["xsrf_token"] = xsrf
        if bl and bl != CONFIG.get("gemini_bl"):
            CONFIG["gemini_bl"] = bl
        if xsrf:
            _persist_auth_to_file(xsrf, bl)
            log("Auth refreshed from Gemini page")
            return True
        log("Auth refresh found no token")
        return False
    except Exception as e:
        log(f"Auth refresh failed: {e}")
        return False


def _is_xsrf_error(e: Exception) -> bool:
    """Check whether an upstream error is a 400 xsrf rejection."""
    import urllib.error as _urlerr
    if isinstance(e, _urlerr.HTTPError) and e.code == 400:
        try:
            return "xsrf" in e.read().decode("utf-8", errors="replace")
        except Exception:
            return True
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 400:
        return True
    return False


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


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


def upstream_echo(raw: str):
    """Return (label, family, variant) echoed by upstream, or None.

    The StreamGenerate response echoes the model that actually served the
    request; comparing it against the requested family/variant detects
    ignored model selection (e.g. expired model ticket).
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

    When a ticket is used it wins over the body fields, so expectations are
    read from the ticket's embedded (family, variant).
    """
    echo = upstream_echo(raw)
    if not echo:
        return
    _, fam, var = echo
    if ticket:
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
            f"the model ticket in CONFIG['model_tickets'] may be expired — "
            f"refresh it from a fresh browser capture")


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, ticket: str = None) -> str:
    """Non-streaming generation with retry."""
    refreshed = False
    if load_cookie()[0] and not CONFIG.get("xsrf_token"):
        refreshed = refresh_auth()
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
    url = _get_url()
    headers = _build_headers(ticket)
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    for attempt in range(CONFIG["retry_attempts"]):
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
            check_routing(raw, model_id, extra_fields, ticket)
            return extract_response_text(raw)
        except Exception as e:
            last_err = e
            if not refreshed and _is_xsrf_error(e) and refresh_auth():
                refreshed = True
                log("Retrying with refreshed auth...")
                body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
                url = _get_url()
                headers = _build_headers(ticket)
            elif attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, ticket: str = None):
    """Streaming generation via httpx with retry on connection failure."""
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields, ticket)
        if text:
            yield text
        return

    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
    url = _get_url()
    headers = _build_headers(ticket)
    client = _get_httpx_client()

    last_err = None
    emitted_raw_text = ""
    refreshed = False
    if load_cookie()[0] and not CONFIG.get("xsrf_token"):
        refreshed = refresh_auth()
        body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
        url = _get_url()
        headers = _build_headers(ticket)
    for attempt in range(CONFIG["retry_attempts"]):
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
                        for t in _extract_texts_from_line(line):
                            if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                continue
                            if not t.startswith(emitted_raw_text):
                                raise RuntimeError("Gemini stream content changed during retry")
                            delta = clean_text(t[len(emitted_raw_text):], strip=False)
                            emitted_raw_text = t
                            if delta:
                                yield delta
            return
        except Exception as e:
            last_err = e
            if not refreshed and _is_xsrf_error(e) and refresh_auth():
                refreshed = True
                log("Stream retry with refreshed auth...")
                body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
                url = _get_url()
                headers = _build_headers(ticket)
            elif attempt < CONFIG["retry_attempts"] - 1:
                log(f"Stream retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
