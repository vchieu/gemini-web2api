"""Model definitions and mapping from Gemini frontend StreamGenerate payloads."""

# Model selection uses TWO fields in the f.req inner array (decoded from live
# browser captures, Sep 2026):
#   inner[79] = family: 1=flash, 3=pro, 4=auto, 5=dynamic-thinking, 6=flash-lite
#   inner[80] = variant: 1=standard, 2=extended/thinking
# E.g. 3.1 Pro=(3,1), 3.1 Pro Extended=(3,2), Flash Extended=(1,2),
# Flash-Lite=(6,1), Flash-Lite Extended=(6,2).
# The proxy previously sent only inner[79] and left inner[80] unset, so the
# server fell back to its default (observed: 3.1 Pro) regardless of request.
# Note: no field selects the exact 3.x point version within a family; the
# server picks its current default (e.g. requesting "3.5-flash" yields (1,1)).

MODELS = {
    "gemini-3.8-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "desc": "Latest workhorse model, best reasoning & coding (Sep 2026)",
    },
    "gemini-3.8-flash-thinking": {
        "mode": 1, "think": 0, "variant": 2,
        "desc": "Deep thinking mode on the latest Flash backend",
    },
    "gemini-3.7-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "desc": "All-around model (Gemini 3.7 Flash)",
    },
    "gemini-3.6-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.5-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "desc": "All-around model (Gemini 3.5 Flash)",
    },
    "gemini-3.5-flash-lite": {
        "mode": 6, "think": 4, "variant": 1,
        "desc": "Cost-efficient high-capacity model (Gemini 3.5 Flash-Lite)",
    },
    "gemini-3.1-flash-lite": {
        "mode": 6, "think": 4, "variant": 1,
        "desc": "Cost-efficient high-capacity model (Gemini 3.1 Flash-Lite)",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 1, "think": 0, "variant": 2,
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4, "variant": 1,
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-3.1-pro-enhanced": {
        "mode": 3, "think": 4, "extra": {31: 2, 80: 3},
        "desc": "Pro with enhanced output (experimental)",
    },
    "gemini-auto": {
        "mode": 4, "think": 4, "variant": 1,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 0, "variant": 2,
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4, "variant": 1,
        "desc": "Lightweight fast model",
    },
}


def resolve_model(model_name: str, default: str = "gemini-3.6-flash"):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields).

    Unknown model names fall back to default rather than erroring,
    since upstream clients may request arbitrary model identifiers.
    """
    think_override = None
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
    if "variant" in cfg and 80 not in extra:
        extra[80] = cfg["variant"]
    return model_name, mode_id, think_mode, None, extra or None