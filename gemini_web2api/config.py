"""Configuration management."""
import json
import os

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.6-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
    # Per-model upstream tickets (X-Goog-Ext-525001261-Jspb). The browser mints
    # one per model family and the server routes BY TICKET, ignoring the
    # f.req [79]/[80] fields when it is absent (falls back to account default).
    # Tickets carry embedded timestamps and expire; refresh by copying the
    # header value from a fresh browser StreamGenerate request
    # (DevTools -> Copy as cURL) into the matching key ("flash"/"pro"/...).
    # The proxy logs a routing-mismatch warning when a ticket stops working.
    "model_tickets": {
        "flash": '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,1,1,"561701FD-A2E8-4275-98B0-636DFE1554F9",null,null,[[6,908199999],[1789884088,624000000]]]',
        "pro": '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,3,1,"32C786FF-9AE2-49E5-A67C-0A35421F63A6",null,null,[[6,620699999],[1789897904,515000000]]]',
        "lite": '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,6,1,"32C786FF-9AE2-49E5-A67C-0A35421F63A6",null,null,[[null,95100000],[1789898801,131000000]]]',
        "flash-thinking": '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,1,2,"279B5F21-C196-4B10-8EC7-31625C0CABE6",null,null,[[null,332100000],[1789899397,281000000]]]',
        "lite-thinking": '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,6,2,"279B5F21-C196-4B10-8EC7-31625C0CABE6",null,null,[[2,950300000],[1789899759,320000000]]]',
        "pro-thinking": '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,3,2,"279B5F21-C196-4B10-8EC7-31625C0CABE6",null,null,[[null,85000000],[1789900208,389000000]]]',
    },
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
