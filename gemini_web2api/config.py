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
