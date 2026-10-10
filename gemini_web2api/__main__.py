"""Entry point: python -m gemini_web2api"""
import argparse
import os
import sys
import uuid

from .config import CONFIG, load_config, find_config
from .models import MODELS, ticket_warnings
from .gemini import HAS_HTTPX, fetch_latest_bl, log
from .server import GeminiHandler, ThreadedServer
from . import __version__

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(host) -> bool:
    """True when the bind address cannot be reached from another machine."""
    if not isinstance(host, str) or not host:
        return False
    value = host.strip().lower().strip("[]")
    return value in _LOOPBACK_HOSTS or value.startswith("127.")


def _guard_bind(host, allow_insecure: bool) -> None:
    """Refuse an exposed bind with no API keys unless explicitly opted in.

    Binding a routable interface with ``api_keys: []`` hands your Google
    session to anyone who can reach the port. With no keys configured and
    no ``--allow-insecure``, we generate a one-time random key at startup,
    print it in the banner, and require it for every request -- so the
    Dockerfile can omit ``--allow-insecure``.  Anyone really wanting to
    expose the port passes ``--allow-insecure`` intentionally.
    """
    if allow_insecure:
        return
    if _is_loopback(host):
        return
    if CONFIG.get("api_keys"):
        return

    # Generate a one-time key and require it; Docker users read it from
    # the startup log and use it with Authorization: Bearer <key>. Written to
    # stderr directly, not through log(): with log_requests=false the key
    # would otherwise be generated and required but never shown.
    key = uuid.uuid4().hex[:32] + uuid.uuid4().hex[:32]
    sys.stderr.write(f"Auto-generated API key (print this): {key}\n")
    sys.stderr.flush()
    CONFIG["api_keys"] = [key]


def _maybe_refresh_bl() -> None:
    """Refresh the Gemini build label; stale labels make every request fail.

    Skipped when the operator pinned ``gemini_bl`` via ``auto_update_bl=false``
    -- otherwise startup silently overwrites whatever they configured.
    """
    if not CONFIG.get("auto_update_bl", True):
        return
    new_bl = fetch_latest_bl()
    if new_bl:
        CONFIG["gemini_bl"] = new_bl


def main():
    parser = argparse.ArgumentParser(description="Gemini Web to OpenAI API")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cookie-file", type=str, default=None)
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help="start even when binding a non-loopback host with no API keys",
    )
    parser.add_argument("--version", action="version", version=f"gemini-web2api {__version__}")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("GEMINI_WEB2API_CONFIG") or find_config()
    if config_path:
        load_config(config_path)

    if args.host:
        CONFIG["host"] = args.host
    if args.port:
        CONFIG["port"] = args.port
    if args.cookie_file:
        CONFIG["cookie_file"] = args.cookie_file
    if args.proxy:
        CONFIG["proxy"] = args.proxy

    # Refresh the Gemini build label; stale labels make every request fail.
    _maybe_refresh_bl()

    port = CONFIG["port"]
    host = CONFIG["host"]
    _guard_bind(host, args.allow_insecure)
    server = ThreadedServer((host, port), GeminiHandler)
    print(f"gemini-web2api v{__version__}")
    print(f"  Listening: http://{host}:{port}")
    print(f"  Base URL:  http://localhost:{port}/v1")
    print(f"  Models:    {', '.join(MODELS.keys())}")
    auth_enabled = bool(CONFIG.get('api_keys'))
    auth_msg = 'enabled' if auth_enabled else 'DISABLED (any client can call this)'
    if auth_enabled and not CONFIG.get('api_keys'):
        # Auto-generated key case: show that auth is required
        auth_msg = 'enabled (auto-generated key required)'
    print(f"  Auth:      {auth_msg}")
    print(f"  Cookie:    {'yes' if CONFIG.get('cookie_file') else 'none (anonymous)'}")
    print(f"  Proxy:     {CONFIG.get('proxy') or 'system env'}")
    print(f"  Streaming: {'httpx (true streaming)' if HAS_HTTPX else 'urllib (buffered)'}")
    print(f"  BL:        {CONFIG['gemini_bl']}"
          f"{'' if CONFIG.get('auto_update_bl', True) else ' (pinned)'}")
    print(f"  Temporary: {'yes' if CONFIG.get('temporary_chats', False) else 'no'}")
    print()
    for warning in ticket_warnings():
        sys.stderr.write(f"WARNING: {warning}\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
