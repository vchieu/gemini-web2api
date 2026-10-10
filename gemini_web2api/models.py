"""Model definitions and mapping from Gemini frontend StreamGenerate payloads."""
import json

# Model selection uses TWO fields in the f.req inner array (decoded from live
# browser captures, Sep 2026):
#   inner[79] = family: 1=flash, 3=pro, 4=auto, 5=dynamic-thinking, 6=flash-lite
#   inner[80] = variant: 1=standard, 2=extended/thinking, 3=enhanced
# E.g. 3.1 Pro=(3,1), Flash Extended=(1,2), Flash-Lite=(6,1).
#
# HOWEVER: the server only honors those fields when the request also carries
# the per-model ticket header X-Goog-Ext-525001261-Jspb (minted by the browser
# per model family; embeds family/variant in plaintext). Without the ticket the
# server ignores [79]/[80] entirely and answers with the account default model
# -- observed as 3.1 Pro, so every requested model silently became Pro. The
# ticket wins over the body fields when both are present, and it is what the
# echo check compares against (see gemini.check_routing).
#
# Note: no field selects the exact 3.x point version within a family; the
# server picks its current default (e.g. requesting "3.5-flash" yields 3.6
# Flash). The "3.x" in a model name is a label, not a routing hint.

# MODE_CATEGORY enum from 028-6eb337387583.js:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

TICKET_HEADER = "X-Goog-Ext-525001261-Jspb"

# `variant` maps to inner[80]; `ticket` is the key into
# CONFIG["model_tickets"]. Models without a ticket have no live-verified
# ticket value, so they fall back to the body fields (and to the account
# default when the upstream declines those too).
MODELS = {
    "gemini-3.7-flash": {
        "mode": 1, "think": 4, "variant": 1, "ticket": "flash",
        "desc": "Latest all-around model (Gemini 3.7 Flash)",
    },
    "gemini-3.6-flash": {
        "mode": 1, "think": 4, "variant": 1, "ticket": "flash",
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.5-flash": {
        "mode": 1, "think": 4, "variant": 1, "ticket": "flash",
        "desc": "Alias for gemini-3.6-flash (backend upgraded)",
    },
    "gemini-3.5-flash-thinking": {
        # think=1 (Extended), not 0: live checks showed 1 produces the longer
        # thinking answer, 0 the shorter one.
        "mode": 2, "think": 1, "variant": 2, "ticket": "flash-thinking",
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4, "variant": 1, "ticket": "pro",
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-3.1-pro-enhanced": {
        # No ticket: a ticket wins over the body, which would throw away the
        # enhanced variant encoded in extra[80]=3.
        "mode": 3, "think": 4, "extra": {31: 2, 80: 3},
        "desc": "Pro with enhanced output (experimental)",
    },
    "gemini-auto": {
        # No ticket: the upstream picks the model for this family itself.
        "mode": 4, "think": 4, "variant": 1,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        # The lite-thinking ticket routes to Flash-Lite Extended (6,2) even
        # though the body says family 5 -- the ticket wins.
        "mode": 5, "think": 1, "variant": 2, "ticket": "lite-thinking",
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4, "variant": 1, "ticket": "lite",
        "desc": "Lightweight fast model",
    },
}


# The (family, variant) each ticket key is expected to embed at ticket
# indexes 14/15. When a configured ticket carries anything else the upstream
# silently serves a different model (today: flash-thinking runs Flash-Lite
# Extended), so startup warns instead of mis-routing quietly. Tickets cannot
# be re-captured programmatically -- the owner refreshes them from a fresh
# browser session.
EXPECTED_TICKET_ROUTES = {
    "flash": (1, 1),
    "flash-thinking": (1, 2),
    "lite": (6, 1),
    "lite-thinking": (6, 2),
    "pro": (3, 1),
    "pro-thinking": (3, 2),
}


def ticket_warnings() -> list:
    """Warning strings for configured tickets embedding an unexpected route.

    Each ``CONFIG["model_tickets"]`` value is parsed and its indexes 14/15
    compared against :data:`EXPECTED_TICKET_ROUTES`. Missing keys are skipped
    (no ticket degrades to the body fields, which ``check_routing`` covers);
    unparsable ones warn. Silent when everything matches.
    """
    from .config import CONFIG
    warnings = []
    tickets = CONFIG.get("model_tickets") or {}
    for key, (want_fam, want_var) in EXPECTED_TICKET_ROUTES.items():
        raw = tickets.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            fam, var = parsed[14], parsed[15]
        except (json.JSONDecodeError, IndexError, TypeError):
            warnings.append(
                f"model ticket '{key}' could not be parsed -- refresh it from "
                f"a fresh browser capture")
            continue
        if fam != want_fam or var != want_var:
            warnings.append(
                f"model ticket '{key}' embeds (family={fam}, variant={var}), "
                f"expected (family={want_fam}, variant={want_var}) -- the "
                f"upstream serves a different model until this ticket is "
                f"refreshed from a fresh browser capture")
    return warnings


def ticket_for(model_name: str):
    """Return the upstream ticket header value for a model, or None.

    Tickets live in ``CONFIG["model_tickets"]`` because they are minted by a
    real browser session and expire: an absent or stale key degrades to the
    body fields rather than erroring, and ``gemini.check_routing`` warns when
    the upstream then answers with a different model.
    """
    from .config import CONFIG
    cfg = MODELS.get(model_name) or {}
    key = cfg.get("ticket")
    if not key:
        return None
    return (CONFIG.get("model_tickets") or {}).get(key)


def resolve_model(model_name: str, default: str = "gemini-3.6-flash"):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields).

    Unknown model names fall back to default rather than erroring,
    since upstream clients may request arbitrary model identifiers.
    """
    think_override = None
    if default not in MODELS:
        default = "gemini-3.6-flash"
    if not isinstance(model_name, str) or not model_name:
        model_name = default
    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None
    cfg = MODELS.get(model_name)
    if not cfg:
        from .gemini import log
        log(f"Unknown model '{model_name}', falling back to '{default}'")
        model_name = default
        cfg = MODELS[default]
    mode_id = cfg["mode"]
    think_mode = think_override if think_override is not None else cfg["think"]
    extra = dict(cfg.get("extra") or {})
    # inner[80] (variant) travels through extra_fields so a model can pin it
    # explicitly -- pro-enhanced already does; never overwrite that.
    if "variant" in cfg and 80 not in extra:
        extra[80] = cfg["variant"]
    return model_name, mode_id, think_mode, None, extra or None
