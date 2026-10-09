"""HTTP server: OpenAI-compatible API endpoints."""
import json
import time
import uuid
import re
import hmac
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .config import CONFIG
from .models import MODELS, resolve_model, ticket_for
from .gemini import generate, generate_stream, log, GeminiError
from .tools import (
    messages_to_prompt,
    parse_tool_calls,
    google_contents_to_prompt,
    parse_google_function_calls,
    build_response_format_instruction,
    strip_code_fence,
    tool_names,
    tool_parameters,
    tool_required_params,
    missing_required_params,
    is_required_tool_choice,
    looks_like_missed_tool_call,
    looks_like_upstream_error,
    pending_tool_request,
)
from .multimodal import detect_image_mime, fetch_image_bytes, upload_image
from . import __version__


ERR_INVALID_REQUEST = "invalid_request_error"
ERR_RATE_LIMIT = "rate_limit_error"
ERR_API = "api_error"

_MISSING = object()


def _estimate_tokens(text: str) -> int:
    return len(text or "") // 4


def _usage(prompt: str, text: str) -> dict:
    p = _estimate_tokens(prompt)
    c = _estimate_tokens(text)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _normalize_stop(stop) -> list:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str) and s]
    return []


def _apply_stop(text: str, stops: list) -> str:
    if not text or not stops:
        return text
    cut = None
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return text[:cut] if cut is not None else text


def _apply_max_tokens(text: str, max_tokens) -> tuple:
    """Return (text, truncated) after applying an estimated token budget."""
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        return text, False
    limit = max_tokens * 4
    if len(text) > limit:
        return text[:limit], True
    return text, False


def _responses_text_format(text_field) -> dict:
    """Map a Responses API ``text.format`` object to a Chat ``response_format``.

    The field name differs (``text.format`` vs ``response_format``) and the
    JSON schema sits one level further up:
    ``{"type":"json_schema","schema":{...}}`` instead of
    ``{"type":"json_schema","json_schema":{"schema":{...}}}``.
    """
    if not isinstance(text_field, dict):
        return None
    fmt = text_field.get("format")
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") == "json_object":
        return {"type": "json_object"}
    if fmt.get("type") == "json_schema":
        if isinstance(fmt.get("json_schema"), dict):
            return {"type": "json_schema", "json_schema": fmt["json_schema"]}
        return {"type": "json_schema", "json_schema": {"schema": fmt.get("schema", {})}}
    return None


def _map_upstream_error(e) -> tuple:
    """Map an upstream exception to (status, message, type, code).

    503 rather than 502: the spec declares 400/401/403/404/429/500/503 for
    ``/chat/completions`` and 400/404/429/503 for ``/responses``. A 502 is
    documented only for the audio endpoints, and a strict client that treats
    undeclared statuses as a protocol error would reject it. Upstream Gemini
    failing is exactly the "service unavailable" case 503 describes.
    """
    status = getattr(e, "status", None)
    if status is None:
        status = getattr(e, "code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 429:
        return 429, f"upstream rate limited: {e}", ERR_RATE_LIMIT, "rate_limit_exceeded"
    if status:
        return 503, f"upstream error ({status}): {e}", ERR_API, None
    return 503, f"upstream error: {e}", ERR_API, None


def _map_google_error(e) -> tuple:
    """Map an upstream exception to (status, message, google_status)."""
    status, message, _type, _code = _map_upstream_error(e)
    gstatus = "RESOURCE_EXHAUSTED" if status == 429 else "UNAVAILABLE"
    return status, message, gstatus


def _upload_images(images: list) -> list:
    """Upload images and return list of file references. Returns None if no images."""
    if not images:
        return None
    file_refs = []
    for item in images:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        data, mime = item
        if isinstance(data, str):
            data = fetch_image_bytes(data)
            mime = mime or "image/png"
        if not data:
            raise RuntimeError("image fetch failed")
        mime = detect_image_mime(data, mime or "image/png")
        try:
            ref = upload_image(data, "image.png", mime or "image/png")
            file_refs.append(ref)
        except Exception as e:
            raise RuntimeError(f"image upload failed: {e}") from e
    return file_refs if file_refs else None


def _tool_retry_attempts(required_tool: bool, tools_active: bool) -> int:
    """Total upstream attempts for a turn that is expected to end in a call.

    One retry leaves a model that answers in prose instead of emitting the
    block failing on a meaningful share of turns (and every ``tool_choice:
    "required"`` failure is a 503 for the client), so the number of extra
    attempts is configurable -- ``tool_retry_attempts`` in the config. The
    default of 1 keeps the cost at "one extra call, and only when the first
    one already failed"; an operator running a forgetful model raises it
    instead of everyone paying for three calls on turns that succeeded.
    """
    if not (required_tool or (tools_active and CONFIG.get("tool_retry_on_miss"))):
        return 1
    try:
        extra = int(CONFIG.get("tool_retry_attempts", 1))
    except (TypeError, ValueError):
        extra = 1
    return 1 + max(0, extra)


def _log_tool_trace(raw_text: str, tool_calls) -> None:
    """Log what the model emitted and what was parsed out of it.

    A tool call that reaches the client with missing or unusable arguments is
    indistinguishable, from the client's side, from a model that simply wrote
    ``{"arguments": {}}``. The raw block is the only place that shows whether
    the model or the parser is at fault, so log it whenever tools are involved.
    """
    blocks = raw_text.count("```tool_call")
    log(f"upstream: {len(raw_text)} chars, {blocks} tool_call marker(s)")
    for m in re.finditer(r"```tool_call[ \t]*\n?(.*?)\n?```", raw_text, re.DOTALL):
        log(f"tool_call block: {m.group(1).strip()[:500]}")
    if blocks and not tool_calls:
        log("tool_call marker(s) present but no tool call was parsed")
    if tool_calls:
        log(f"parsed tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")
    if not tool_calls:
        # With tools offered and nothing parsed, the raw text is the only place
        # that distinguishes "model answered in prose" from "model used a fence
        # we do not recognise" (```json / ```function_call / bare JSON) -- both
        # look identical in the marker count above.
        fences = sorted({(m.group(1) or "bare").lower()
                         for m in re.finditer(r"```([A-Za-z0-9_+-]*)", raw_text)})
        log(f"no tool_call parsed; fences seen: {', '.join(fences) or 'none'}")
        log(f"no tool_call parsed; text head: {raw_text[:400]!r}")


class GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        client_ip = self.client_address[0] if self.client_address else "-"
        log(f"{client_ip} {fmt % args}")

    # ─── CORS / security helpers ──────────────────────────────────────────────

    def _cors_origin(self) -> str:
        """Return the Access-Control-Allow-Origin value, or '' to omit.

        When api_keys is empty the server is unauthenticated; sending
        ``*`` would let any malicious page read the response.  We only
        emit the header when the operator explicitly configured
        ``cors_origins``.
        """
        origins = CONFIG.get("cors_origins") or []
        if not origins:
            return ""
        if "*" in origins:
            return "*"
        # Echo back the requesting origin if it is in the whitelist.
        origin = self.headers.get("Origin", "")
        if origin in origins:
            return origin
        return ""

    def _send_cors_headers(self):
        """Send CORS headers if applicable."""
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type, x-api-key, x-goog-api-key",
            )

    def _check_host(self) -> bool:
        """Reject requests whose Host header is not a loopback / configured host.

        This blocks DNS-rebinding attacks where an attacker's domain resolves
        to 127.0.0.1 but the browser sends ``Host: attacker.com``.

        Three cases where the check is deliberately not restrictive:

        * ``api_keys`` configured -- the request already had to present a valid
          key, so a rebound hostname buys nothing (``do_GET`` never runs this
          check either, so only POST was ever gated);
        * a wildcard bind (``0.0.0.0`` / ``::``) -- the operator chose to serve
          every interface, so LAN IPs and container hostnames must pass, not
          just ``localhost``;
        * no ``Host`` header at all (non-HTTP/1.1 edge case).

        The hostname is parsed with ``urlsplit`` so ``[::1]:8081`` yields
        ``::1`` instead of the unbracketed ``[::1]`` that ``rsplit(":")``
        produced, and a port is never mistaken for part of the name.
        """
        if CONFIG.get("api_keys"):
            return True
        host = self.headers.get("Host", "")
        if not host:
            return True
        hostname = (urllib.parse.urlsplit("//" + host).hostname or "").lower()
        if not hostname:
            return True
        allowed = {"localhost", "127.0.0.1", "::1"}
        bind_host = (CONFIG.get("host") or "").lower().strip("[]")
        if bind_host:
            if bind_host in ("0.0.0.0", "::", "*"):
                return True  # Wildcard bind: reachable on every interface.
            allowed.add(bind_host)
        return hostname in allowed

    def _check_origin(self) -> bool:
        """Reject cross-origin requests when no API key is configured.

        A malicious page can issue a ``text/plain`` POST (simple request, no
        preflight) to our endpoint.  Without an API key we cannot distinguish
        a legitimate local client from an attacker's page, so we reject any
        request that carries an ``Origin`` header not in the whitelist.
        """
        keys = CONFIG.get("api_keys") or []
        if keys:
            return True  # Authenticated: origin check is not needed
        origin = self.headers.get("Origin")
        if not origin:
            return True  # No Origin: likely a non-browser client
        origins = CONFIG.get("cors_origins") or []
        if "*" in origins:
            return True
        return origin in origins

    # ─── response helpers ─────────────────────────────────────────────────────

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_api_error(self, status, message, type_=ERR_API, code=None):
        """Send an OpenAI-shaped error body."""
        self.send_json(
            {"error": {"message": message, "type": type_, "param": None, "code": code}},
            status,
        )

    def send_google_error(self, status, message, gstatus="UNKNOWN"):
        """Send a Google Gemini-shaped error body."""
        self.send_json({"error": {"code": status, "message": message, "status": gstatus}}, status)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._send_cors_headers()
        self.end_headers()

    def _sse_chunk(self, cid, model, delta, finish_reason, include_usage=False):
        # ``logprobs`` is required (nullable) on the choice of a chat chunk and
        # ``delta`` must open with an empty ``content`` so clients that read
        # ``delta.content`` on the first chunk (NextChat among them) see a
        # string rather than an absent key.
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "logprobs": None,
                         "finish_reason": finish_reason}],
        }
        if include_usage:
            # stream_options.include_usage: every chunk carries a usage field,
            # null until the final usage chunk.
            chunk["usage"] = None
        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _sse_usage(self, cid, model, prompt, text):
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": _usage(prompt, text),
        }
        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _sse_event_error(self, status, message, type_=ERR_API, code=None):
        payload = {"error": {"message": message, "type": type_, "param": None, "code": code}}
        self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _sse_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

# ─── request parsing / auth / routing ─────────────────────────────────────

    def _parse_body(self, body: bytes):
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _read_request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in transfer_encoding.lower():
            chunks = []
            total = 0
            limit = CONFIG.get("max_body_bytes", 32 * 1024 * 1024)
            while True:
                size_line = self.rfile.readline()
                if not size_line:
                    break
                size_text = size_line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError:
                    raise ValueError("invalid chunked request body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                total += size
                if total > limit:
                    raise ValueError("request body too large")
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)

        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return b""
        try:
            length = int(raw_len)
        except (ValueError, TypeError):
            raise ValueError("invalid Content-Length header")
        if length < 0:
            raise ValueError("invalid Content-Length header")
        if length > CONFIG.get("max_body_bytes", 32 * 1024 * 1024):
            raise ValueError("request body too large")
        return self.rfile.read(length) if length else b""

    def _route_path(self) -> str:
        """Return the request path without query string / trailing slash."""
        return urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

    def _authorized(self):
        keys = CONFIG.get("api_keys") or []
        if not keys:
            return True
        provided = []
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            provided.append(auth[7:])
        for header in ("x-api-key", "x-goog-api-key"):
            value = self.headers.get(header)
            if value:
                provided.append(value)
        query = urllib.parse.urlparse(self.path).query
        for pair in query.split("&"):
            if pair.startswith("key="):
                provided.append(urllib.parse.unquote(pair[4:]))
        for value in provided:
            for key in keys:
                if isinstance(key, str) and hmac.compare_digest(value.encode(), key.encode()):
                    return True
        return False

    def _needs_auth(self, path: str) -> bool:
        return path not in ("/", "/health")

    def do_OPTIONS(self):
        requested = self.headers.get("Access-Control-Request-Headers")
        self.send_response(204)
        self._send_cors_headers()
        self.send_header(
            "Access-Control-Allow-Headers",
            requested or "Authorization, Content-Type, x-api-key, x-goog-api-key",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        try:
            path = self._route_path()
            if self._needs_auth(path) and not self._authorized():
                self.send_api_error(401, "invalid api key", ERR_INVALID_REQUEST, "invalid_api_key")
                return
            if path in ("/v1/models", "/models"):
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in MODELS.items()
                ]})
            elif path.startswith("/v1/models/") or path.startswith("/models/"):
                model_id = path.rsplit("/", 1)[-1]
                cfg = MODELS.get(model_id)
                if cfg:
                    self.send_json({"id": model_id, "object": "model", "created": 1700000000,
                                    "owned_by": "google", "description": cfg["desc"]})
                else:
                    self.send_api_error(404, f"model '{model_id}' not found",
                                        ERR_INVALID_REQUEST, "model_not_found")
            elif path.startswith("/v1beta/models"):
                self.send_json({"models": [
                    {"name": f"models/{n}", "displayName": n, "description": c["desc"],
                     "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]}
                    for n, c in MODELS.items()
                ]})
            elif re.match(r"^/(v1/)?responses/[^/]+$", path):
                # The spec declares GET /v1/responses/{id} (200/404) but this
                # server keeps no store, so every id is unknown. Deeper paths
                # (e.g. /input_items) fall through to the generic 404.
                self.send_api_error(404, f"response '{path.rsplit('/', 1)[-1]}' not found",
                                    ERR_INVALID_REQUEST, "response_not_found")
            elif path in ("/", "/health"):
                self.send_json({"status": "ok", "version": __version__,
                                "models": list(MODELS.keys())})
            else:
                self.send_api_error(404, f"not found: {path}", ERR_INVALID_REQUEST)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            # Without this an exception escapes and the socket is dropped with
            # no response at all, leaving the client waiting on a dead socket.
            log(f"GET error: {e}")
            try:
                self.send_api_error(500, str(e), ERR_API)
            except Exception:
                pass

    def _check_security(self) -> str:
        """Run Host / Origin / Content-Type checks. Returns '' or an error message."""
        if not self._check_host():
            return "forbidden host"
        if not self._check_origin():
            return "cross-origin request not allowed"
        return ""

    def _resolve_request_model(self, req):
        """Resolve the requested model. Returns a tuple or None (error sent)."""
        raw = req.get("model")
        base = raw.split("@think=")[0] if isinstance(raw, str) else raw
        if CONFIG.get("strict_models") and isinstance(base, str) and base and base not in MODELS:
            self.send_api_error(404, f"model '{raw}' not found",
                                ERR_INVALID_REQUEST, "model_not_found")
            return None
        model_name, model_id, think_mode, err, extra = resolve_model(
            raw or CONFIG["default_model"])
        if err:
            self.send_api_error(400, err, ERR_INVALID_REQUEST)
            return None
        echo = raw if isinstance(raw, str) and raw else model_name
        # The ticket is what upstream routes on; [79]/[80] alone are ignored.
        return echo, model_name, model_id, think_mode, extra, ticket_for(model_name)

    def do_POST(self):
        try:
            path = self._route_path()
            if self._needs_auth(path) and not self._authorized():
                self.send_api_error(401, "invalid api key", ERR_INVALID_REQUEST, "invalid_api_key")
                return
            # Security checks: Host, Origin, Content-Type
            security_err = self._check_security()
            if security_err:
                self.send_api_error(403, security_err, ERR_INVALID_REQUEST)
                return
            # Require application/json for POST requests (except OPTIONS)
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self.send_api_error(415, "Content-Type must be application/json", ERR_INVALID_REQUEST)
                return
            body = self._read_request_body()
            if path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat(body)
            elif path in ("/v1/responses", "/responses"):
                self._handle_responses(body)
            elif ":streamGenerateContent" in path:
                self._handle_google_generate(body, stream=True)
            elif ":generateContent" in path:
                self._handle_google_generate(body, stream=False)
            else:
                self.send_api_error(404, f"not found: {path}", ERR_INVALID_REQUEST)
        except ValueError as e:
            # Body parsing errors: invalid Content-Length, too large, etc.
            if "too large" in str(e):
                self.send_api_error(413, str(e), ERR_INVALID_REQUEST)
            else:
                self.send_api_error(400, str(e), ERR_INVALID_REQUEST)
            return
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_api_error(500, str(e), ERR_API)
            except Exception:
                pass

    def _unsupported_method(self):
        """Answer PUT/PATCH/DELETE with a JSON error instead of 501 HTML.

        ``BaseHTTPRequestHandler`` replies to any method it has no ``do_*``
        for with a bare-text ``501``, which no OpenAI SDK can parse.
        """
        try:
            path = self._route_path()
            if self._needs_auth(path) and not self._authorized():
                self.send_api_error(401, "invalid api key", ERR_INVALID_REQUEST,
                                    "invalid_api_key")
                return
            # Drain the body before answering. Left unread it becomes the next
            # "request line", and the resulting connection reset can discard
            # the very response written here.
            try:
                self._read_request_body()
            except ValueError as e:
                if "too large" in str(e):
                    self.send_api_error(413, str(e), ERR_INVALID_REQUEST)
                    return
            # DELETE /v1/responses/{id} is a real spec endpoint (200 or 404),
            # and this server stores nothing, so every id is simply unknown.
            if self.command == "DELETE" and re.match(r"^/(v1/)?responses/[^/]+$", path):
                self.send_api_error(404, f"response '{path.rsplit('/', 1)[-1]}' not found",
                                    ERR_INVALID_REQUEST, "response_not_found")
                return
            self.send_api_error(405, f"{self.command} is not allowed for {path}",
                                ERR_INVALID_REQUEST, "method_not_allowed")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"{self.command} error: {e}")
            try:
                self.send_api_error(500, str(e), ERR_API)
            except Exception:
                pass

    def do_PUT(self):
        self._unsupported_method()

    def do_PATCH(self):
        self._unsupported_method()

    def do_DELETE(self):
        self._unsupported_method()

    def _generate_with_tool_retry(self, prompt, model_id, think_mode, file_refs,
                                  extra, tools_active, allowed_names,
                                  tool_schemas, required_tool, messages=None,
                                  ticket=None, tool_required=None):
        """Run ``generate()``, retrying while the reply did not use a tool.

        Five shapes all arrive as a normal HTTP 200 yet leave the client
        believing the turn is over:

        * an empty body -- a connection that died mid-generation leaves
          nothing to answer with, and 200-with-no-content reads as "done";
        * the upstream's canned failure sentence ("I encountered an error doing
          what you asked...") -- nothing was generated, a retry usually works,
          and passing it on reads as a finished answer;
        * the model announcing an action ("let me read...") without ever
          emitting the block;
        * the model *fabricating* an answer about a file no tool call has
          opened -- the reply is fluent and complete, so nothing in the text
          itself looks wrong (see ``pending_tool_request``);
        * ``tool_choice: "required"`` with no block at all -- or a block that
          parses but omits a required parameter, which fails on the client's
          side and costs a round-trip before anyone learns what was missing.

        The retry appends a hard instruction. The number of attempts comes from
        :func:`_tool_retry_attempts`. Raises once attempts are exhausted (or if
        ``generate`` itself fails on the final attempt) so the caller decides
        how the failure is reported -- both endpoints map it through
        ``_map_upstream_error``.
        """
        retry_on_miss = bool(CONFIG.get("tool_retry_on_miss"))
        attempts = _tool_retry_attempts(required_tool, tools_active)
        # Computed once: it inspects the whole conversation, not the reply.
        owed = pending_tool_request(messages) if tools_active else None

        text, tool_calls, missing = "", None, []
        for attempt in range(attempts):
            call_prompt = prompt
            if attempt > 0:
                # "tool_call block ONLY" is right when the client demands a
                # call, but on a heuristic retry it would rewrite a finished
                # summary into a call nobody asked for -- and an agent happily
                # runs that, then asks again. Conditional wording still pushes
                # an acting model to act while letting a genuine answer stand.
                nudge = (
                    "IMPORTANT: Respond with a tool_call block ONLY."
                    if required_tool else
                    "IMPORTANT: Do not describe a tool call -- make it. If "
                    "answering requires reading or changing something, call "
                    "the tool now; otherwise give your final answer."
                )
                if missing:
                    # A model that omitted a parameter believes its call is
                    # complete, so it has to be told which fields were absent.
                    nudge += (f" The previous tool call was missing required "
                              f"parameter(s): {', '.join(missing)} -- include "
                              "every required parameter.")
                    missing = []
                call_prompt = prompt + "\n\n" + nudge
            try:
                raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Tool retry after upstream error: {e}")
                    continue
                raise
            text, tool_calls = raw, None
            if tools_active and text:
                text, tool_calls = parse_tool_calls(text, allowed_names, tool_schemas)
                _log_tool_trace(raw, tool_calls)
            if tool_calls:
                missing = missing_required_params(tool_calls, tool_required)
                if missing:
                    # Always logged: even when no attempt is left this is the
                    # only place that shows the model wrote a call the client
                    # cannot run.
                    log(f"tool_call missing required parameter(s): "
                        f"{', '.join(missing)}")
                    if attempt + 1 < attempts:
                        log(f"Tool retry (attempt {attempt + 1}/{attempts}): "
                            "asking the model to fill them in")
                        continue
                break
            if required_tool:
                continue
            if attempt + 1 >= attempts:
                break
            if not (text or "").strip():
                # An upstream that produced nothing (observed once as an empty
                # body after a dropped connection) is never a usable answer:
                # handing it to the client as 200 says "finished, nothing to
                # do" and Opencode stops the turn there.
                log(f"Tool retry (attempt {attempt + 1}/{attempts}): upstream "
                    "returned an empty response")
                continue
            if looks_like_upstream_error(text):
                log(f"Tool retry (attempt {attempt + 1}/{attempts}): upstream "
                    "returned its canned error placeholder instead of a response")
                continue
            if retry_on_miss and owed:
                log(f"Tool retry (attempt {attempt + 1}/{attempts}): {owed}")
                continue
            if retry_on_miss and tools_active and looks_like_missed_tool_call(text):
                log(f"Tool retry (attempt {attempt + 1}/{attempts}): response "
                    f"looks like a missed tool call ({len(text)} chars)")
                continue
            break

        # The canned sentence is an upstream failure, not something the model
        # said. Handing it to the client as 200 says "answered" when it was not.
        if not tool_calls and looks_like_upstream_error(text):
            raise GeminiError("upstream returned an error placeholder instead "
                              "of a response; please retry")
        return text, tool_calls

    # ─── /v1/chat/completions ─────────────────────────────────────────────────

    def _handle_chat(self, body: bytes):
        req = self._parse_body(body)
        if not isinstance(req, dict):
            self.send_api_error(400, "invalid JSON body", ERR_INVALID_REQUEST)
            return
        resolved = self._resolve_request_model(req)
        if resolved is None:
            return
        echo_model, model_name, model_id, think_mode, extra, ticket = resolved

        n = req.get("n", 1)
        if isinstance(n, int) and n > 1:
            self.send_api_error(400, "n>1 is not supported", ERR_INVALID_REQUEST, "unsupported_value")
            return

        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        messages = req.get("messages", [])
        if not isinstance(messages, list):
            self.send_api_error(400, "messages must be a list", ERR_INVALID_REQUEST)
            return
        prompt, images = messages_to_prompt(messages, tools, tool_choice)

        rf_instruction = build_response_format_instruction(req.get("response_format"))
        if rf_instruction:
            prompt = prompt + rf_instruction
        if not prompt.strip():
            self.send_api_error(400, "empty prompt", ERR_INVALID_REQUEST)
            return

        stream = bool(req.get("stream", False))
        stop_strings = _normalize_stop(req.get("stop"))
        max_tokens = req.get("max_tokens")
        if max_tokens is None:
            max_tokens = req.get("max_completion_tokens")
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_api_error(503, f"upstream error: {e}", ERR_API)
            return

        allowed_names = tool_names(tools) or None
        tool_schemas = tool_parameters(tools) or None
        tool_required = tool_required_params(tools) or None

        # Pure streaming (no tools, no post-processing): stream tokens straight through.
        # When response_format is set the text still has to be defenced and
        # normalised, so it has to go through the buffered path below.
        if stream and not rf_instruction and (not tools or tool_choice == "none"):
            self._stream_chat(cid, echo_model, prompt, model_id, think_mode, file_refs,
                              extra, stop_strings, max_tokens, include_usage, ticket)
            return

        # Tool calls need the full text; retry once when the reply did not use
        # one (see _generate_with_tool_retry).
        required_tool = is_required_tool_choice(tool_choice)
        tools_active = bool(tools) and tool_choice != "none"
        try:
            text, tool_calls = self._generate_with_tool_retry(
                prompt, model_id, think_mode, file_refs, extra,
                tools_active, allowed_names, tool_schemas, required_tool,
                messages=messages, ticket=ticket, tool_required=tool_required)
        except Exception as e:
            self.send_api_error(*_map_upstream_error(e))
            return

        if rf_instruction and text:
            text = strip_code_fence(text)
        text = _apply_stop(text, stop_strings)
        text, truncated = _apply_max_tokens(text, max_tokens)

        if not text and not tool_calls:
            self.send_api_error(503, "empty response from upstream", ERR_API)
            return

        # `refusal` and `logprobs` are required-nullable on the spec's
        # response message/choice; strict SDKs (Go, Rust, Java) reject an
        # absent key where Python's only checks for null.
        msg = {"role": "assistant", "content": text or None, "refusal": None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else ("length" if truncated else "stop")

        if stream:
            try:
                self._start_sse()
                self._sse_chunk(cid, echo_model, {"role": "assistant", "content": ""},
                                None, include_usage)
                if text:
                    self._sse_chunk(cid, echo_model, {"content": text}, None, include_usage)
                for index, tc in enumerate(tool_calls or []):
                    self._sse_chunk(cid, echo_model,
                                    {"tool_calls": [{"index": index, **tc}]},
                                    None, include_usage)
                self._sse_chunk(cid, echo_model, {}, finish, include_usage)
                if include_usage:
                    self._sse_usage(cid, echo_model, prompt, text)
                self._sse_done()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_json({
            "id": cid, "object": "chat.completion", "created": int(time.time()),
            "model": echo_model,
            "choices": [{"index": 0, "message": msg, "logprobs": None,
                         "finish_reason": finish}],
            "usage": _usage(prompt, text),
        })

    @staticmethod
    def _iter_with_first(first, gen):
        yield first
        for delta in gen:
            yield delta

    def _stream_chat(self, cid, model, prompt, model_id, think_mode, file_refs,
                     extra, stop_strings, max_tokens, include_usage, ticket=None):
        try:
            gen = generate_stream(prompt, model_id, think_mode, file_refs, extra, ticket)
            first = next(gen, _MISSING)
        except Exception as e:
            self.send_api_error(*_map_upstream_error(e))
            return
        if first is _MISSING:
            self.send_api_error(503, "empty response from upstream", ERR_API)
            return

        self._start_sse()
        self._sse_chunk(cid, model, {"role": "assistant", "content": ""},
                        None, include_usage)
        full_text = ""
        finish = "stop"
        hold = max((len(s) for s in stop_strings), default=0)
        buf = ""
        try:
            for raw in self._iter_with_first(first, gen):
                if not raw:
                    continue
                buf += raw
                hit = None
                for s in stop_strings:
                    idx = buf.find(s)
                    if idx != -1 and (hit is None or idx < hit):
                        hit = idx
                if hit is not None:
                    piece = buf[:hit]
                    if piece:
                        full_text += piece
                        self._sse_chunk(cid, model, {"content": piece}, None, include_usage)
                    buf = ""
                    break
                # `>` rather than `>=`: a reply that lands exactly on the budget
                # is complete, and only a truncated reply may claim `length`.
                if max_tokens and (len(full_text) + len(buf)) > max_tokens * 4:
                    allowed = max(0, max_tokens * 4 - len(full_text))
                    piece = buf[:allowed]
                    if piece:
                        full_text += piece
                        self._sse_chunk(cid, model, {"content": piece}, None, include_usage)
                    buf = ""
                    finish = "length"
                    break
                if len(buf) > hold:
                    piece = buf[:len(buf) - hold] if hold else buf
                    buf = buf[-hold:] if hold else ""
                    if piece:
                        full_text += piece
                        self._sse_chunk(cid, model, {"content": piece}, None, include_usage)
            else:
                if buf:
                    full_text += buf
                    self._sse_chunk(cid, model, {"content": buf}, None, include_usage)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            log(f"Stream error: {e}")
            try:
                self._sse_event_error(*_map_upstream_error(e))
                self._sse_done()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        self._sse_chunk(cid, model, {}, finish, include_usage)
        if include_usage:
            self._sse_usage(cid, model, prompt, full_text)
        self._sse_done()

# ─── /v1/responses (Codex CLI) ───────────────────────────────────────────

    def _normalize_responses_tools(self, tools):
        if not tools:
            return None
        normalized = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                fn = tool.get("function", tool)
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                if not name:
                    log("Responses: ignoring function tool without a name")
                    continue
                normalized.append({"type": "function", "function": {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }})
            else:
                log(f"Responses: ignoring unsupported tool type '{tool.get('type')}'")
        return normalized or None

    def _responses_messages(self, input_items, instructions):
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
            return messages
        if not isinstance(input_items, list):
            return messages
        for item in input_items:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call_output":
                output = item.get("output", "")
                if isinstance(output, list):
                    output = "\n".join(
                        c.get("text", "") for c in output
                        if isinstance(c, dict) and c.get("type") in ("output_text", "text")
                    )
                messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                 "content": output})
            elif itype == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                # Codex omits `arguments` on a call it wants replayed from context;
                # an empty string would serialise into the broken `"arguments": }`.
                call = {"id": call_id, "type": "function",
                        "function": {"name": item.get("name", ""),
                                     "arguments": item.get("arguments") or "{}"}}
                if messages and messages[-1].get("role") == "assistant" \
                        and messages[-1].get("tool_calls"):
                    messages[-1]["tool_calls"].append(call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif itype in ("input_text", "input_image", "image"):
                messages.append({"role": "user", "content": [item]})
            elif item.get("role") == "assistant":
                content_parts = item.get("content", [])
                text_acc, tc_list = "", []
                if isinstance(content_parts, list):
                    for c in content_parts:
                        if isinstance(c, dict):
                            if c.get("type") == "output_text":
                                text_acc += c.get("text", "")
                            elif c.get("type") == "function_call":
                                tc_list.append(c)
                elif isinstance(content_parts, str):
                    text_acc = content_parts
                message = {"role": "assistant", "content": text_acc or None}
                if tc_list:
                    message["tool_calls"] = [
                        {"id": tc.get("call_id", f"call_{i}"), "type": "function",
                         "function": {"name": tc.get("name", ""),
                                      "arguments": tc.get("arguments") or "{}"}}
                        for i, tc in enumerate(tc_list)
                    ]
                messages.append(message)
            else:
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                messages.append({"role": role, "content": item.get("content", "")})
        return messages

    def _handle_responses(self, body: bytes):
        req = self._parse_body(body)
        if not isinstance(req, dict):
            self.send_api_error(400, "invalid JSON body", ERR_INVALID_REQUEST)
            return
        resolved = self._resolve_request_model(req)
        if resolved is None:
            return
        echo_model, model_name, model_id, think_mode, extra, ticket = resolved

        tools = self._normalize_responses_tools(req.get("tools"))
        messages = self._responses_messages(req.get("input", []), req.get("instructions"))
        # `null` is a legal JSON value here; the spec has no null variant, so
        # fall back to "auto" rather than echoing a null tool_choice back.
        tool_choice = req.get("tool_choice") or "auto"
        prompt, images = messages_to_prompt(messages, tools, tool_choice)

        rf = req.get("response_format")
        if rf is None:
            rf = _responses_text_format(req.get("text"))
        rf_instruction = build_response_format_instruction(rf)
        if rf_instruction:
            prompt = prompt + rf_instruction
        if not prompt.strip():
            self.send_api_error(400, "empty input", ERR_INVALID_REQUEST)
            return

        stop_strings = _normalize_stop(req.get("stop"))
        max_tokens = req.get("max_output_tokens")

        allowed_names = tool_names(tools) or None
        tool_schemas = tool_parameters(tools) or None
        tool_required = tool_required_params(tools) or None
        required_tool = is_required_tool_choice(tool_choice)
        tools_active = bool(tools) and tool_choice != "none"

        try:
            file_refs = _upload_images(images)
            text, tool_calls = self._generate_with_tool_retry(
                prompt, model_id, think_mode, file_refs, extra,
                tools_active, allowed_names, tool_schemas, required_tool,
                messages=messages, ticket=ticket, tool_required=tool_required)
        except Exception as e:
            self.send_api_error(*_map_upstream_error(e))
            return
        if rf_instruction and text:
            text = strip_code_fence(text)
        text = _apply_stop(text, stop_strings)
        text, truncated = _apply_max_tokens(text, max_tokens)
        if not text and not tool_calls:
            self.send_api_error(503, "empty response from upstream", ERR_API)
            return

        # Responses reports a cut-off answer as incomplete rather than completed.
        final_status = "incomplete" if truncated else "completed"
        # `incomplete_details` is required on the object and null when the
        # answer is complete; it must not disappear from the payload.
        incomplete_details = {"reason": "max_output_tokens"} if truncated else None
        # A message that max_output_tokens cut off is itself incomplete --
        # leaving it "completed" while the response says "incomplete" is the
        # contradiction strict clients flag.
        message_status = "incomplete" if truncated else "completed"

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"],
                               "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or not tool_calls:
            output.append({"type": "message", "id": mid, "role": "assistant",
                           "status": message_status,
                           "content": [{"type": "output_text", "text": text or "",
                                        "annotations": [], "logprobs": []}]})

        usage = {
            "input_tokens": len(prompt) // 4,
            "output_tokens": len(text or "") // 4,
            "total_tokens": (len(prompt) + len(text or "")) // 4,
            # Required breakdowns: their absence fails strict SDKs even though
            # the totals are all we can actually measure.
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

        # Fields the spec marks required on every response object. Request
        # fields are echoed back unchanged where we honour them, otherwise the
        # spec's own default is used.
        created_at = int(time.time())
        instructions = req.get("instructions")
        tools_echo = req.get("tools")
        metadata = req.get("metadata")
        parallel = req.get("parallel_tool_calls")
        temperature = req.get("temperature")
        top_p = req.get("top_p")

        def _number(value, default):
            # `bool` is an ``int`` subclass: `temperature: true` must not be
            # echoed back as a number the schema rejects.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return default
            return value

        def response_fields(status, items, token_usage):
            return {
                "id": rid,
                "object": "response",
                "created_at": created_at,
                "status": status,
                "model": echo_model,
                "output": items,
                "error": None,
                "incomplete_details": incomplete_details,
                "instructions": instructions if isinstance(instructions, str) else None,
                "tools": tools_echo if isinstance(tools_echo, list) else [],
                "parallel_tool_calls": parallel if isinstance(parallel, bool) else True,
                "metadata": metadata if isinstance(metadata, dict) else None,
                "tool_choice": tool_choice,
                "temperature": _number(temperature, 1),
                "top_p": _number(top_p, 1),
                "access_programs": None,
                "usage": token_usage,
            }

        if req.get("stream"):
            self._start_sse()
            sequence_number = 0

            def emit(event_type, **fields):
                nonlocal sequence_number
                sequence_number += 1
                event = {"type": event_type, "sequence_number": sequence_number, **fields}
                self.wfile.write(f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode())

            emit("response.created", response=response_fields("in_progress", [], None))
            emit("response.in_progress", response=response_fields("in_progress", [], None))
            for output_index, item in enumerate(output):
                if item["type"] == "function_call":
                    pending_item = {
                        "type": "function_call", "id": item["id"], "call_id": item["call_id"],
                        "name": item["name"], "arguments": "", "status": "in_progress",
                    }
                    emit("response.output_item.added", output_index=output_index, item=pending_item)
                    emit("response.function_call_arguments.delta", item_id=item["id"],
                         output_index=output_index, delta=item["arguments"])
                    emit("response.function_call_arguments.done", item_id=item["id"],
                         output_index=output_index, arguments=item["arguments"])
                    emit("response.output_item.done", output_index=output_index, item=item)
                elif item["type"] == "message":
                    pending_item = {
                        "type": "message", "id": item["id"], "role": "assistant",
                        "status": "in_progress", "content": [],
                    }
                    emit("response.output_item.added", output_index=output_index, item=pending_item)
                    for content_index, content_part in enumerate(item["content"]):
                        event_fields = {"item_id": item["id"], "output_index": output_index,
                                        "content_index": content_index}
                        emit("response.content_part.added", **event_fields, part={
                            "type": "output_text", "text": "", "annotations": [],
                            "logprobs": []})
                        # `logprobs` is required on both text events (an empty
                        # list: we never see upstream probabilities).
                        emit("response.output_text.delta", **event_fields,
                             delta=content_part["text"], logprobs=[])
                        emit("response.output_text.done", **event_fields,
                             text=content_part["text"], logprobs=[])
                        emit("response.content_part.done", **event_fields, part=content_part)
                    emit("response.output_item.done", output_index=output_index, item=item)
            final_event = "response.incomplete" if truncated else "response.completed"
            emit(final_event, response=response_fields(final_status, output, usage))
            self.wfile.flush()
        else:
            self.send_json(response_fields(final_status, output, usage))

    # ─── /v1beta/models (Google Gemini CLI) ──────────────────────────────────

    def _handle_google_generate(self, body: bytes, stream: bool):
        req = self._parse_body(body)
        if not isinstance(req, dict):
            self.send_google_error(400, "invalid JSON", "INVALID_ARGUMENT")
            return
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        model_name = m.group(1) if m else CONFIG["default_model"]
        model_name, model_id, think_mode, err, extra = resolve_model(model_name)
        if err:
            self.send_google_error(400, err, "INVALID_ARGUMENT")
            return
        ticket = ticket_for(model_name)

        tool_config = req.get("toolConfig", {})
        fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")
        has_tools = bool(req.get("tools")) and fc_mode != "NONE"
        prompt, images = google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_google_error(400, "empty content", "INVALID_ARGUMENT")
            return

        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_google_error(502, f"upstream error: {e}", "UNAVAILABLE")
            return
        log(f"Google API: model={model_name} stream={stream} tools={has_tools} prompt_len={len(prompt)}")

        if stream and not has_tools:
            try:
                gen = generate_stream(prompt, model_id, think_mode, file_refs, extra, ticket)
                first = next(gen, _MISSING)
            except Exception as e:
                self.send_google_error(*_map_google_error(e))
                return
            if first is _MISSING:
                self.send_google_error(502, "empty response from upstream", "UNAVAILABLE")
                return
            try:
                self._start_sse()
                full_text = ""
                for delta in self._iter_with_first(first, gen):
                    if not delta:
                        continue
                    full_text += delta
                    chunk_obj = {
                        "candidates": [{"content": {"parts": [{"text": delta}],
                                                    "role": "model"}, "index": 0}],
                        "modelVersion": model_name,
                    }
                    self.wfile.write(
                        f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                final_chunk = {
                    "candidates": [{"finishReason": "STOP", "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": len(prompt) // 4,
                        "candidatesTokenCount": len(full_text) // 4,
                        "totalTokenCount": (len(prompt) + len(full_text)) // 4,
                    },
                    "modelVersion": model_name,
                }
                self.wfile.write(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Google stream error: {e}")
                # The 200 is already sent, so the only way to tell the client
                # is a Google-shaped error event on the same stream.
                status, message, gstatus = _map_google_error(e)
                try:
                    err_obj = {"error": {"code": status, "message": message, "status": gstatus}}
                    self.wfile.write(f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                close = getattr(gen, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            return

        # The Google path used to call generate() once and hand back whatever
        # came out, so none of the "the model said it would act but did not"
        # protections the OpenAI endpoints have applied here: a client using
        # the native API was the one most likely to be left with prose.
        required_tool = fc_mode == "ANY"
        retry_on_miss = bool(CONFIG.get("tool_retry_on_miss"))
        attempts = _tool_retry_attempts(required_tool, has_tools)

        text, clean_text, function_calls = "", "", []
        for attempt in range(attempts):
            call_prompt = prompt
            if attempt > 0:
                nudge = (
                    "IMPORTANT: Respond with a function_call block ONLY."
                    if required_tool else
                    "IMPORTANT: Do not describe a function call -- make it. If "
                    "answering requires reading or changing something, call "
                    "the tool now; otherwise give your final answer."
                )
                call_prompt = prompt + "\n\n" + nudge
            try:
                raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Google tool retry after upstream error: {e}")
                    continue
                raise
            text = raw
            clean_text, function_calls = "", []
            if has_tools and raw:
                clean_text, function_calls = parse_google_function_calls(raw)
            if function_calls:
                break
            if attempt + 1 >= attempts:
                break
            if not (raw or "").strip():
                log(f"Google tool retry (attempt {attempt + 1}/{attempts}): "
                    "upstream returned an empty response")
                continue
            if looks_like_upstream_error(raw):
                log(f"Google tool retry (attempt {attempt + 1}/{attempts}): "
                    "upstream returned its canned error placeholder instead of "
                    "a response")
                continue
            if required_tool:
                continue
            if retry_on_miss and looks_like_missed_tool_call(raw):
                log(f"Google tool retry (attempt {attempt + 1}/{attempts}): "
                    f"response looks like a missed tool call ({len(raw)} chars)")
                continue
            break

        if not text:
            self.send_google_error(502, "empty response from upstream", "UNAVAILABLE")
            return

        # The canned sentence is an upstream failure, not something the model
        # said. Returning it as a candidate tells the client "answered".
        if looks_like_upstream_error(text):
            status, message, gstatus = _map_google_error(GeminiError(
                "upstream returned an error placeholder instead of a response; "
                "please retry"))
            self.send_google_error(status, message, gstatus)
            return

        response_parts = []
        if has_tools and function_calls:
            if clean_text:
                response_parts.append({"text": clean_text})
            for fc in function_calls:
                response_parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})
        else:
            response_parts.append({"text": text})

        candidate = {
            "content": {"parts": response_parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text) // 4,
            "totalTokenCount": (len(prompt) + len(text)) // 4,
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self._start_sse()
            self.wfile.write(f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True