"""Configuration management."""
import json
import os

DEFAULT_CONFIG = {
    "port": 8081,
    # Bind to loopback by default: without API keys, anyone on the same network
    # could otherwise use your Google session. Set to "0.0.0.0" behind a
    # container / reverse proxy that provides its own access control.
    "host": "127.0.0.1",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    # Refresh gemini_bl from gemini.google.com on startup. Set to false to pin
    # the value above -- otherwise startup overwrites whatever you configured.
    "auto_update_bl": True,
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.6-flash",
    # When False (default) unknown model names silently fall back to
    # default_model. When True they return 404 model_not_found.
    "strict_models": False,
    "log_requests": True,
    # Optional file sink for the same lines `log()` prints on stderr. Relative
    # paths resolve against the server's working directory.
    "log_file": None,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    # Per-model upstream tickets (X-Goog-Ext-525001261-Jspb). The browser mints
    # one per model family and the upstream routes BY TICKET: without it the
    # f.req [79]/[80] fields are ignored and the account default model answers
    # (observed: 3.1 Pro for every requested model). Tickets embed their
    # (family, variant) in plaintext and expire, so refresh a key by copying
    # the header value from a fresh browser StreamGenerate request (DevTools ->
    # Copy as cURL) for the matching family ("flash", "pro", "lite", ...).
    # gemini.check_routing logs a mismatch warning when one stops working.
    "model_tickets": {},
    # CORS origins allowed to read the response. Empty list means no CORS
    # headers are emitted at all, which is the safe default when api_keys is
    # also empty -- a malicious page could otherwise fetch the local server.
    # Set to ["*"] explicitly when you have api_keys configured and want to
    # allow any browser origin.
    "cors_origins": [],
    # Hard cap on a single request body. A large base64 image can otherwise
    # eat unbounded RAM while the server reads it.
    "max_body_bytes": 32 * 1024 * 1024,
    "temporary_chats": False,
    # Retry once when a response with tools offered looks like the model talked
    # about acting instead of acting (or came back as the upstream's canned
    # failure sentence). One extra upstream call at most, and off entirely if
    # the heuristic proves noisy.
    "tool_retry_on_miss": True,
    # Extra upstream attempts (beyond the first) when a turn is expected to end
    # in a tool call: tool_choice "required", or tools offered with
    # tool_retry_on_miss above. 0 restores the old single call, 2 or 3 helps a
    # model that keeps answering in prose instead of emitting the block -- each
    # extra attempt only ever costs a call that was already failing, never one
    # that succeeded.
    "tool_retry_attempts": 1,
    # Upper bound for the assembled prompt (bytes) sent upstream. Agent
    # sessions (opencode/cursor) send tens of KB per file read; at 60000 a
    # single read of a 1000-line file filled the budget, the joiner kept only
    # that message's tail (dropping its "[Tool result ...]" label and the
    # pending question), and the model answered from memory. Live probes
    # against the real upstream accepted prompts well past 500 KB, so the
    # default is much larger; lower it only to shrink request payloads.
    "prompt_max_bytes": 262144,
    # Diagnostics: dump every raw upstream response to debug_raw_file so the
    # payload shape (thinking vs answer vs tool call) can be inspected. Off by
    # default -- raw responses can be large and carry conversation content.
    "debug_raw": False,
    "debug_raw_file": "gemini-raw.log",
    # Four-leg raw trace for deep investigation: CLIENT -> PROXY (do_POST body),
    # PROXY -> MODEL (built prompt), MODEL -> PROXY (raw upstream bytes) and
    # PROXY -> CLIENT (JSON/SSE frames), appended to trace_file. Gated
    # separately from debug_raw because it also captures client conversation
    # content on both ends.
    "debug_trace": False,
    "trace_file": "gemini-trace.log",
}

CONFIG = dict(DEFAULT_CONFIG)


def load_config(path: str = None):
    """Load config from JSON file."""
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
    return CONFIG


def find_config():
    """Search for config file in standard locations."""
    for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None
