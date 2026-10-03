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
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.6-flash",
    # When False (default) unknown model names silently fall back to
    # default_model. When True they return 404 model_not_found.
    "strict_models": False,
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
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
