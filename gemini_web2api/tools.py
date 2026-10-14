"""Tool calling and multimodal message parsing."""
import ast
import json
import re
import uuid
import base64
import binascii
import unicodedata
from urllib.parse import unquote_to_bytes

# Fallback bound for the generated prompt when config carries no
# ``prompt_max_bytes``. Keeps large tool lists / long histories from being
# rejected by the upstream web endpoint. The live default lives in
# config.DEFAULT_CONFIG (well above this): 60000 was a single file read away
# from exhaustion in agent sessions, which cost the pending question.
PROMPT_MAX_BYTES = 60000


def _prompt_max_bytes() -> int:
    """Effective prompt budget in bytes: config override, else the fallback.

    Read from CONFIG at call time (not import time) so a config reload or a
    test patching ``CONFIG["prompt_max_bytes"]`` takes effect immediately.
    """
    from .config import CONFIG
    try:
        value = int(CONFIG.get("prompt_max_bytes", PROMPT_MAX_BYTES))
    except (TypeError, ValueError):
        return PROMPT_MAX_BYTES
    return value if value > 0 else PROMPT_MAX_BYTES


def _build_tool_choice_instruction(tool_choice, tool_defs: list) -> str:
    """Build tool_choice constraint instruction.

    tool_choice values:
      - "none": do not call any tool
      - "auto": decide whether to call tools (default)
      - "required": must call at least one tool
      - {"type": "function", "function": {"name": "xxx"}}: must call specific tool
    """
    if tool_choice == "none":
        return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
    if tool_choice == "required":
        return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
    if isinstance(tool_choice, dict):
        # Chat Completions uses {"type":"function","function":{"name":"x"}};
        # the Responses API drops the nesting and uses {"type":"function","name":"x"}.
        fn = tool_choice.get("function")
        fn_name = fn.get("name", "") if isinstance(fn, dict) else ""
        if not fn_name:
            fn_name = tool_choice.get("name") or ""
        if fn_name:
            return f'\n\nIMPORTANT: You MUST call the tool "{fn_name}". Do not call other tools.'
    return ""


def _decode_data_url(url: str):
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
    if not match:
        return None
    mime = match.group(1) or "image/png"
    is_base64 = bool(match.group(2))
    data = match.group(3)
    try:
        if is_base64:
            return base64.b64decode(data, validate=True), mime
        return unquote_to_bytes(data), mime
    except (ValueError, TypeError, binascii.Error):
        return None


def _image_from_url(url: str, mime: str = None):
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        return _decode_data_url(url)
    return url, mime or "image/png"


def _image_from_part(part: dict):
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url", {})
        if isinstance(image_url, dict):
            return _image_from_url(image_url.get("url"), image_url.get("mime_type"))
        return _image_from_url(image_url)
    if part_type in ("input_image", "image"):
        image_url = part.get("image_url") or part.get("url")
        if isinstance(image_url, dict):
            return _image_from_url(image_url.get("url"), image_url.get("mime_type"))
        if image_url:
            return _image_from_url(image_url, part.get("mime_type"))
        image_data = part.get("data") or part.get("base64")
        if isinstance(image_data, str):
            mime = part.get("mime_type") or part.get("media_type") or "image/png"
            if image_data.startswith("data:"):
                return _decode_data_url(image_data)
            try:
                return base64.b64decode(image_data, validate=True), mime
            except (ValueError, TypeError, binascii.Error):
                return None
    return None


def _stringify_content(content) -> str:
    """Flatten OpenAI message content (str or list of parts) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") in ("text", "input_text", "output_text"):
                    out.append(c.get("text", ""))
                elif isinstance(c.get("text"), str):
                    out.append(c["text"])
            elif isinstance(c, str):
                out.append(c)
        return "\n".join(p for p in out if p)
    return str(content)


def tool_names(tools) -> set:
    """Collect declared function names from an OpenAI tools list."""
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if tool.get("type") == "function" else tool
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
    return names


def tool_parameters(tools) -> dict:
    """Map tool name -> set of top-level parameter names from a tools list.

    Used to recover a call whose model omitted ``name``: the parameter keys are
    matched against these sets, so a guess is only ever made when exactly one
    declared tool could have produced them.
    """
    schemas = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if tool.get("type") == "function" else tool
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        params = fn.get("parameters") or tool.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        schemas[fn["name"]] = set(props) if isinstance(props, dict) else set()
    return schemas


def tool_required_params(tools) -> dict:
    """Map tool name -> set of required parameter names from a tools list.

    Complements :func:`tool_parameters`: a call can carry every declared
    property and still be unusable because the one required one is missing,
    and neither the client nor the model learns anything from the failure it
    produces.
    """
    required = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if tool.get("type") == "function" else tool
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        params = fn.get("parameters") or tool.get("parameters") or {}
        names = params.get("required") if isinstance(params, dict) else None
        if isinstance(names, list):
            required[fn["name"]] = {n for n in names if isinstance(n, str)}
    return required


def missing_required_params(tool_calls, required) -> list:
    """``["tool.param", ...]`` for every required parameter a call left out.

    A call that parses but omits a required parameter reaches the client as a
    perfectly valid tool call, fails on the client's side, and the error the
    client sends back costs a round-trip the model spends guessing what was
    wrong. Reported per *tool.parameter* so the retry instruction can name the
    exact thing to add. Arguments that are not a JSON object are skipped: they
    cannot be checked, and they already carry their own "unusable arguments"
    report.
    """
    missing = []
    if not required:
        return missing
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        params = required.get(name)
        if not params:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}", strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        missing.extend(f"{name}.{p}" for p in sorted(params) if p not in args)
    return missing


# Sits between the dropped leading block and the kept tail so the model can see
# that history was removed rather than silently wondering what it missed.
_TRUNCATION_MARKER = "[...truncated...]"


def _utf8_safe_slice(data: bytes, start: int, end: int) -> bytes:
    """Slice bytes, snapping to valid UTF-8 character boundaries.

    Raw byte slicing cuts multi-byte characters in half; ``decode``
    with ``errors="ignore"`` then silently drops the fragments.
    Snapping to boundaries keeps whole characters intact.
    """
    if start >= end:
        return b""
    # Advance past any partial continuation byte at the start.
    while start < end and (data[start] & 0xC0) == 0x80:
        start += 1
    # Find the first byte of the character that ends at end-1.
    pos = end - 1
    while pos > start and (data[pos] & 0xC0) == 0x80:
        pos -= 1
    first = data[pos]
    if first & 0xE0 == 0xC0:
        char_len = 2
    elif first & 0xF0 == 0xE0:
        char_len = 3
    elif first & 0xF8 == 0xF0:
        char_len = 4
    else:
        char_len = 1
    # If the character extends past end, it is cut off: drop it.
    if pos + char_len > end:
        end = pos
    return data[start:end]


def _squeeze_part(part: str, cap: int) -> str:
    """Reduce one message to ~``cap`` bytes, keeping both its ends.

    A combined part starts with the ``[Assistant]:`` / ``[Tool result for
    read ...]`` labels that say *what* this blob is and ends with the newest
    lines of the file or a trailing question. Keeping only the tail (the old
    behaviour) handed the model an unattributed slab of code: it could not
    tell which file it came from, re-read files it had already read, and --
    once older messages were evicted too -- answered with no idea what the
    task was. Head keeps the labels, tail keeps the question/last lines, and
    the marker in between shows content was dropped.
    """
    raw = part.encode("utf-8")
    if len(raw) <= cap:
        return part
    marker = _TRUNCATION_MARKER.encode("utf-8")
    # Head 1/3 (labels + opening context), tail the rest (newest content).
    head_len = max(0, cap // 3 - len(marker) - 2)
    tail_len = max(0, cap - head_len - len(marker) - 2)
    if head_len <= 0 or tail_len <= 0:
        # Degenerate cap: fall back to keeping the tail end only, which at
        # least preserves a question asked at the end of the message.
        return _utf8_safe_slice(raw, len(raw) - cap, len(raw)).decode(
            "utf-8", errors="ignore")
    squeezed = (_utf8_safe_slice(raw, 0, head_len) + b"\n" + marker + b"\n"
                + _utf8_safe_slice(raw, len(raw) - tail_len, len(raw)))
    return squeezed.decode("utf-8", errors="ignore")


def _join_prompt_parts(parts: list, max_bytes: int, pinned_head: str = None,
                       pinned_first: str = None) -> str:
    """Join message parts into a prompt, dropping the *middle* when too long.

    Slicing the assembled prompt from the front (the obvious way) keeps the
    system instruction and the oldest history while throwing away the newest
    user message -- i.e. exactly the question that needs answering. Instead we
    keep the leading block (tool definitions / system instruction) and as many
    of the newest messages as fit, and mark the gap.

    ``pinned_head`` is reserved *before* anything else is allocated. Without it
    the newest messages (a large tool result, say) can fill the budget on their
    own, leaving nothing for the leading block -- and a prompt with no tool
    definitions makes the model answer in prose instead of calling a tool,
    which is indistinguishable from the model simply refusing to act.

    ``pinned_first`` is the first user message (the plan/task in agent
    sessions). It gets its own reservation so long middle histories cannot
    evict it: losing the plan while keeping tool definitions produces a model
    that calls tools correctly but for a forgotten task.
    """
    items = [p for p in parts if p]

    def nbytes(text: str) -> int:
        return len(text.encode("utf-8"))

    # Cap the pin so a pathological tool list cannot consume the budget on its
    # own; ~40% leaves the majority for the history and the pending question.
    # Squeezing (head+tail) keeps the header and the trailing constraint;
    # byte-slicing cut the constraint off first. Pins are only squeezed when
    # the prompt actually overflows: squeezing unconditionally would stamp a
    # "[...truncated...]" marker into a prompt that fits untouched.
    pinned = (pinned_head.strip() if pinned_head else "")
    first_pin = (pinned_first.strip() if pinned_first else "")
    first_idx = -1
    if first_pin:
        for i, p in enumerate(items):
            if p == pinned_first:
                first_idx = i
                break

    full = "\n\n".join(([pinned] if pinned else []) + items)
    if nbytes(full) <= max_bytes:
        return full

    if pinned:
        cap = max_bytes * 2 // 5
        if nbytes(pinned) > cap:
            pinned = _squeeze_part(pinned, cap).strip() or pinned
    if first_pin:
        fcap = max(1, max_bytes // 7)
        if nbytes(first_pin) > fcap:
            first_pin = _squeeze_part(first_pin, fcap).strip() or first_pin
        # Squeeze the original in place so the head loop below (which fills
        # from the start in order) can afford it inside its reservation.
        # Order is preserved: system stays before the first user message.
        if first_idx != -1:
            items[first_idx] = first_pin

    # Squeezing the pins alone may have been enough (a huge tool block with
    # a short history): re-check before dropping any middle.
    full = "\n\n".join(([pinned] if pinned else []) + items)
    if nbytes(full) <= max_bytes:
        return full

    from .gemini import log
    log(f"Prompt truncated to {max_bytes} bytes")
    if pinned:
        log(f"Tool block pinned: {nbytes(pinned)} bytes "
            f"({100 * nbytes(pinned) // max_bytes}% of budget)")
    if first_pin:
        log(f"First user message pinned: {nbytes(first_pin)} bytes")

    # Two separators around the marker.
    budget = max_bytes - nbytes(_TRUNCATION_MARKER) - 4
    if pinned:
        budget -= nbytes(pinned) + 2
    if budget <= 0:
        # Degenerate budget (tiny max_bytes): give up on pinning rather than
        # drop the question entirely.
        pinned = ""
        budget = max_bytes - nbytes(_TRUNCATION_MARKER) - 4
    if budget <= 0:
        return full.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")

    # Reserve the first user message's share before the newest messages are
    # allocated: otherwise a giant tail part eats the budget and the plan --
    # the first thing the head loop would need -- never fits. When the first
    # message is also the newest (a short session), no reservation is needed:
    # the tail keeps it naturally.
    skip_first_in_tail = bool(first_pin) and first_idx != -1 \
        and first_idx != len(items) - 1
    first_cost = 0
    if skip_first_in_tail:
        first_cost = nbytes(first_pin) + 2
    tail_budget = budget - first_cost
    if tail_budget < 0:
        tail_budget = 0

    # 1. Keep the newest messages first: they carry the pending question and
    #    the latest tool results. No single message may claim more than half
    #    the budget: when one giant part (a 46 KB file read) was allowed to
    #    take everything, it evicted every older message including the user's
    #    original request, and only its tail -- stripped of the
    #    "[Tool result for read ...]" label at its head -- reached the model.
    #    The pinned first message is skipped here: it is allocated in step 2.
    tail = []
    taken = set()
    used = 0
    part_cap = max(1, budget // 2)
    for pos in range(len(items) - 1, -1, -1):
        if skip_first_in_tail and pos == first_idx:
            continue
        part = items[pos]
        if nbytes(part) > part_cap:
            part = _squeeze_part(part, part_cap)
        cost = nbytes(part) + (2 if tail else 0)
        if used + cost > tail_budget:
            break
        tail.append(part)
        taken.add(pos)
        used += cost
    tail.reverse()

    # 2. Spend whatever is left on the leading block (tool defs, system, ...).
    #    This only ever adds a prefix, so history in the middle stays dropped.
    #    The pinned first message is admitted at its natural position (after
    #    the system prompt, where it already sits); the reservation above
    #    guarantees room for its squeezed form.
    remaining = budget - used
    head = []
    head_used = 0
    for pos, part in enumerate(items):
        if skip_first_in_tail and pos == first_idx:
            part = first_pin
        elif pos in taken:
            break
        cost = nbytes(part) + (2 if head else 0)
        if head_used + cost > remaining:
            break
        head.append(part)
        head_used += cost

    blocks = ([pinned] if pinned else []) + head
    if blocks:
        return "\n\n".join(blocks + [_TRUNCATION_MARKER] + tail)
    return "\n\n".join([_TRUNCATION_MARKER] + tail)


def _compact_tool(t: dict, desc_max: int = 300) -> str:
    """One line per tool: name, parameter names, types and which are required.

    Used when the full JSON schema no longer fits. Dropping the schemas
    outright (the previous behaviour) leaves the model calling a tool whose
    parameters it can no longer see, which produces calls that fail
    validation -- so signatures are always kept and only prose is shortened.
    Nested object/array shapes are flagged so the model knows a parameter
    takes a structured value even though its inner schema was cut.
    """
    p = t.get("parameters") or {}
    props = p.get("properties") if isinstance(p, dict) else None
    props = props if isinstance(props, dict) else {}
    required = set(p.get("required", [])) if isinstance(p, dict) else set()

    def _kind(v) -> str:
        if not isinstance(v, dict):
            return "any"
        kind = v.get("type", "any")
        if kind in ("object", "array") and ("properties" in v or "items" in v):
            return f"{kind}(nested schema omitted)"
        return kind

    args = ", ".join(
        f'{k}{"" if k in required else "?"}: {_kind(v)}'
        for k, v in props.items()
    )
    return f'- {t["name"]}({args}): {(t.get("description") or "")[:desc_max]}'


# Appended after the last message whenever tools are offered. It has to be the
# final thing the model reads: in a long conversation the original instruction
# to *use* the tools sits far back, and the model reverts to answering from
# memory -- which looks exactly like a successful answer with no tool call.
_TOOL_REMINDER = ("\n\n[Reminder: you have working tools. To know what a file "
                  "contains or what a command produced, call the tool -- do not "
                  "answer from memory and do not claim to have read or run "
                  "anything you have not.]")

# Appended instead of _TOOL_REMINDER when the conversation already ends in a
# tool result. The model has just acted and tends to act again (read-read-read
# loops); this tells it the result above is usable and a repeated identical
# call is never the next step.
_TOOL_RESULT_REMINDER = ("\n\n[Reminder: the tool result above is already in "
                         "the conversation. If it is enough, answer now in "
                         "plain text; otherwise call the next tool. Do not "
                         "repeat a call whose result is already present.]")


def _example_call(tool_defs: list) -> str:
    """Render a copy-paste-shaped example call from the first declared tool.

    Placeholders (``func_name``/``param``) taught weak models to emit the
    placeholder itself as the call, so the example names a real tool and real
    parameter keys: one entry per ``required`` parameter (fallback: the first
    two declared properties; ``{}`` when the tool takes none), with values
    typed by the schema type.

    String values carry an ``EXAMPLE_`` prefix so a verbatim copy is
    detectable: :func:`parse_tool_calls` drops calls whose arguments are
    still the placeholder and tells the model to fill in real values.
    """
    first = (tool_defs or [None])[0] or {}
    name = first.get("name") if isinstance(first, dict) else None
    name = name or "tool_name"
    params = first.get("parameters") if isinstance(first, dict) else None
    props = params.get("properties") if isinstance(params, dict) else None
    props = props if isinstance(props, dict) else {}
    required = params.get("required") if isinstance(params, dict) else None
    if isinstance(required, list):
        keys = [k for k in required if k in props]
    else:
        keys = []
    if not keys:
        keys = list(props)[:2]
    example_args = {}
    for key in keys:
        spec = props.get(key) or {}
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind in ("integer", "number"):
            example_args[key] = 0
        elif kind == "boolean":
            example_args[key] = False
        elif kind == "array":
            example_args[key] = []
        elif kind == "object":
            example_args[key] = {}
        else:
            example_args[key] = f"EXAMPLE_{key}"
    return json.dumps({"name": name, "arguments": example_args},
                      ensure_ascii=False)


def _is_placeholder_args(args_json: str) -> bool:
    """True when parsed arguments still carry the ``EXAMPLE_`` placeholder."""
    try:
        args = json.loads(args_json, strict=False)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(args, dict):
        return False
    return any(isinstance(v, str) and v.startswith("EXAMPLE_")
               for v in args.values())


def _tool_use_block(tools_json: str, constraint: str, tool_defs: list = None) -> str:
    """The ``# Tool Use`` header the model must always be able to see.

    Pinned by ``_join_prompt_parts``: without it a large tool result can fill
    the prompt budget on its own and evict the tool definitions entirely, after
    which the model has no idea any tool exists.
    """
    return (
        "# Tool Use\n\n"
        "The tools below are connected to the user's machine and execute for "
        "real the moment you call them. You have full access to them.\n\n"
        "Call format (use this exact format):\n"
        '```tool_call\n'
        f'{_example_call(tool_defs)}\n'
        "```\n\n"
        "When calling tools:\n"
        "- Each turn do exactly ONE of: (A) you need to read or change "
        "something -> output ONLY tool_call block(s); (B) you already have "
        "everything -> answer as plain text with NO block. Never both, and "
        "never any prose after the block.\n"
        "- Before answering a request to review, audit, inspect or find bugs in "
        "source code, you MUST read the actual contents of the relevant files with "
        "a read/search tool first -- reading one file is not reviewing a codebase. "
        "Listing or globbing file names is not reading: "
        "if you have only listed or opened one file, read the other relevant ones before you "
        "answer. If you name files as part of the review, read every one of them -- "
        "a file you only mentioned is a file you have not reviewed.\n"
        "- Fixed flow for EVERY turn, in order, no skipping: "
        "(1) do you already have a [Tool result for ...] message proving "
        "everything you need? If YES -> answer as plain text. "
        "If NO -> output ONLY tool_call block(s) and stop. "
        "A guess, a generic checklist, or instructions telling the user to "
        "do it themselves are never an answer. "
        "(2) When the result arrives, go back to (1).\n"
        "- Output ONLY tool_call block(s): no prose, explanation or apology "
        "before or after them.\n"
        '- Every parameter belongs inside the "arguments" object, keyed exactly '
        'like the "parameters" properties below -- never beside "name".\n'
        "- Emit several blocks in one turn when the calls are independent.\n"
        "- Search results are leads, not evidence: after a glob/grep survey, "
        "read the matched files before answering or editing -- never quote "
        "or patch code you only saw in a snippet.\n"
        "- Before editing a symbol, check its other usages with a search "
        "tool; after editing, re-read the changed lines before reporting "
        "success -- an unverified fix is a claim, not evidence.\n"
        "- When reporting a bug, quote the exact lines verbatim: line "
        "numbers without quotes cannot be checked.\n"
        "- Never state that you have read a file, run a command or inspected "
        "the code unless a [Tool result for ...] message proves it. Knowing a "
        "file's contents or a command's output requires calling the tool.\n"
        "- Never claim the tools are unavailable, restricted, or that you lack "
        "permission: they are connected and will run.\n"
        "- Never write [Tool result for ...] or [Assistant]: yourself, and "
        "never invent a result: emit the block and stop.\n"
        "- Triple backticks inside a string argument (Markdown or code the user "
        "asked you to save) are literal text: keep all three of them. A fence "
        'inside the JSON never closes the "tool_call" block, so shortening '
        "them corrupts the file you were asked to write.\n\n"
        f"Available tools:\n{tools_json}"
        f"{constraint}"
    )


def messages_to_prompt(messages: list, tools: list = None, tool_choice=None) -> tuple:
    """Convert OpenAI messages to (prompt_str, images_list).

    Returns (prompt, images) where images is a list of (bytes, mime_type) tuples.
    """
    from .gemini import log

    max_bytes = _prompt_max_bytes()

    parts = []
    images = []
    tool_block = None

    if tools and tool_choice != "none":
        tool_defs = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", tool) if tool.get("type") == "function" else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", tool.get("name", ""))
            if not name:
                continue
            tool_defs.append({
                "name": name,
                "description": fn.get("description", tool.get("description", "")),
                "parameters": fn.get("parameters", tool.get("parameters", {})),
            })
        if tool_defs:
            constraint = _build_tool_choice_instruction(tool_choice, tool_defs)
            # Use compact JSON by default to save space
            tools_json = json.dumps(tool_defs, ensure_ascii=False)
            tool_bytes = len(tools_json.encode("utf-8"))
            # If still too large, trade description prose for space while
            # keeping every name, parameter, type and required flag: a model
            # that cannot see a parameter calls the tool with the wrong shape.
            if tool_bytes > max_bytes // 3:
                for desc_max in (300, 150, 80):
                    tools_json = "\n".join(_compact_tool(t, desc_max)
                                           for t in tool_defs)
                    tool_bytes = len(tools_json.encode("utf-8"))
                    if tool_bytes <= max_bytes // 3:
                        break
                log(f"Tool definitions compacted to signatures "
                    f"(descriptions cut to {desc_max} chars, {tool_bytes} bytes): "
                    "parameter names and types kept")
            else:
                log(f"Tool definitions: {len(tool_defs)} tools, {tool_bytes} bytes")
            tool_block = _tool_use_block(tools_json, constraint, tool_defs)

    # Map tool_call ids -> function names so tool results can be labelled correctly.
    id_to_name = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    tcid = tc.get("id")
                    fn = tc.get("function") or {}
                    if tcid and isinstance(fn, dict) and fn.get("name"):
                        id_to_name[tcid] = fn["name"]

    # Track which tool result messages have been combined with their assistant tool_call
    combined_tool_result_ids = set()
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role == "developer":
            role = "system"
        content = msg.get("content", "")
        if content is None:
            # An explicit ``content: null`` (legal JSON) must not reach the
            # f-strings below as the literal text "None".
            content = ""

        if isinstance(content, list):
            text_parts = []
            for c in content:
                if not isinstance(c, dict):
                    if isinstance(c, str):
                        text_parts.append(c)
                    continue
                if c.get("type") in ("text", "input_text", "output_text"):
                    text_parts.append(c.get("text", ""))
                    continue
                image = _image_from_part(c)
                if image:
                    images.append(image)
                    text_parts.append("[Image attached]")
                elif isinstance(c.get("text"), str):
                    text_parts.append(c["text"])
            content = " ".join(p for p in text_parts if p)

        if role == "system":
            parts.append(f"[System instruction]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_strs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function") or {}
                    name = json.dumps(fn.get("name", ""), ensure_ascii=False)
                    # An absent/empty "arguments" must still serialise as valid JSON.
                    args = fn.get("arguments") or "{}"
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    tc_strs.append(
                        f'```tool_call\n{{"id": {json.dumps(tc.get("id", ""))}, '
                        f'"name": {name}, "arguments": {args}}}\n```'
                    )
                # Combine with subsequent tool results to keep them together
                combined = f"[Assistant]: {content or ''}\n" + "\n".join(tc_strs)
                for j in range(i + 1, len(messages)):
                    next_msg = messages[j]
                    if not isinstance(next_msg, dict):
                        break
                    if next_msg.get("role") != "tool":
                        break
                    tcid = next_msg.get("tool_call_id", "")
                    combined_tool_result_ids.add(tcid)
                    next_content = next_msg.get("content", "")
                    if isinstance(next_content, list):
                        text_parts = []
                        for c in next_content:
                            if not isinstance(c, dict):
                                if isinstance(c, str):
                                    text_parts.append(c)
                                continue
                            if c.get("type") in ("text", "input_text", "output_text"):
                                text_parts.append(c.get("text", ""))
                                continue
                            image = _image_from_part(c)
                            if image:
                                images.append(image)
                                text_parts.append("[Image attached]")
                            elif isinstance(c.get("text"), str):
                                text_parts.append(c["text"])
                        next_content = " ".join(p for p in text_parts if p)
                    name = next_msg.get("name") or id_to_name.get(tcid) or "tool"
                    body = _stringify_content(next_content)
                    combined += f"\n[Tool result for {name} (id={tcid})]: {body}"
                parts.append(combined)
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            tcid = msg.get("tool_call_id", "")
            # Skip if already combined with assistant tool_call
            if tcid in combined_tool_result_ids:
                continue
            name = msg.get("name") or id_to_name.get(tcid) or "tool"
            body = _stringify_content(content)
            parts.append(f"[Tool result for {name} (id={tcid})]: {body}")
        else:
            parts.append(_stringify_content(content))

    if tool_block:
        last = messages[-1] if messages else None
        last_role = last.get("role") if isinstance(last, dict) else None
        # A turn that already ends in a tool result needs no push to act --
        # it needs permission to stop acting and answer.
        parts.append(_TOOL_RESULT_REMINDER if last_role == "tool"
                     else _TOOL_REMINDER)
    first_user_pin = None
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            first_user_pin = _stringify_content(msg.get("content", ""))
            if first_user_pin and first_user_pin.strip():
                break
            first_user_pin = None
    return _join_prompt_parts(parts, max_bytes, pinned_head=tool_block,
                              pinned_first=first_user_pin), images


# Keys that describe the call itself rather than its parameters. Models emit
# flattened payloads (`{"name": "run_commands", "commands": [...]}`) as readily
# as the canonical nested one, so every other key counts as a parameter.
_CALL_META_KEYS = {"name", "id", "type", "description", "tool", "tool_name"}

# Aliases used for the nested parameter object.
_ARGUMENT_KEYS = ("arguments", "args", "input")

_NOTHING = object()


def extract_arguments(data: dict, props=None):
    """Pull a tool call's parameters out of a parsed payload.

    Handles the canonical ``{"name": ..., "arguments": {...}}`` shape and the
    variants models actually emit: ``args`` / ``input`` aliases, a
    ``parameters`` wrapper, and the flattened form where the parameters sit
    beside ``name``. Returns ``{}`` only when the payload carries none.

    ``props`` is the declared parameter set for the resolved tool (or None
    when unknown): a ``_CALL_META_KEYS`` entry that the schema declares --
    opencode's ``bash`` really takes a ``description`` -- is a parameter, not
    metadata, and survives. Without the schema every meta key is still
    dropped, so a bare ``name``/``id``/``type`` never leaks into arguments.

    A tool that genuinely declares a parameter named ``args``/``input`` (with
    a string value) is not a wrapper: the alias is only honoured when the
    schema does not declare it, or when the payload carries nothing else.
    """
    for key in _ARGUMENT_KEYS:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, dict):
            if props is not None and key in props:
                siblings = set(data) - _CALL_META_KEYS - {key}
                if siblings:
                    continue
            return value
        if isinstance(value, str) and value.strip():
            if props is not None and key in props:
                continue
            return value
    if props is None:
        rest = {k: v for k, v in data.items() if k not in _CALL_META_KEYS}
    else:
        rest = {k: v for k, v in data.items()
                if k not in _CALL_META_KEYS or k in props}
    if len(rest) == 1:
        key, value = next(iter(rest.items()))
        if key in ("parameters", "params") and isinstance(value, dict):
            return value
    return rest or {}


def _arguments_json(args) -> tuple:
    """Serialise a tool call's arguments into a JSON object string.

    Returns ``(json_text, degradation)``; ``degradation`` is None when the
    arguments were usable. ``function.arguments`` has to parse as JSON -- and
    for every OpenAI-style client it has to be an object -- otherwise the
    client rejects the call with a bare "Invalid input" that tells the model
    nothing. A payload that cannot be salvaged therefore degrades to ``{}``,
    a well-formed call the model can retry after seeing the client's error.
    """
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False), None
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return "{}", "empty string"
        parsed = _NOTHING
        strict_ok = True
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            strict_ok = False
            # Same leniency as _iter_tool_blocks: a string argument holding a
            # file body can contain a literal newline. It is re-serialised
            # below because the original text would reach the client with the
            # newline still unescaped, i.e. as invalid JSON.
            try:
                parsed = json.loads(text, strict=False)
            except (json.JSONDecodeError, ValueError):
                parsed = _NOTHING
        if isinstance(parsed, dict):
            if strict_ok:
                return text, None
            return json.dumps(parsed, ensure_ascii=False), None
        # Models occasionally hand back a Python literal instead of JSON.
        try:
            recovered = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            recovered = _NOTHING
        if isinstance(recovered, dict):
            try:
                return json.dumps(recovered, ensure_ascii=False), None
            except (TypeError, ValueError):
                # recovered contains non-JSON-serializable types (set, tuple,
                # bytes, etc.). Convert to a best-effort JSON-safe dict.
                def json_safe(obj):
                    if isinstance(obj, dict):
                        return {str(k): json_safe(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        return [json_safe(v) for v in obj]
                    if isinstance(obj, (str, int, float, bool)) or obj is None:
                        return obj
                    return str(obj)
                return json.dumps(json_safe(recovered), ensure_ascii=False), None
        return "{}", "not a JSON object"
    if args is None:
        return "{}", "missing"
    if isinstance(args, (list, int, float, bool)):
        return json.dumps(args, ensure_ascii=False), None
    return "{}", f"unsupported type {type(args).__name__}"


def _arguments_dict(data: dict, props=None) -> dict:
    """``extract_arguments`` for callers that need a JSON object, always."""
    args = extract_arguments(data, props)
    if isinstance(args, dict):
        return args
    text, _ = _arguments_json(args)
    try:
        parsed = json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    return parsed if isinstance(parsed, dict) else {}


def _infer_tool_name(data: dict, tool_schemas) -> str:
    """Recover the tool name when the model leaves it out.

    The payload's parameter keys are matched against the declared schemas and a
    name is returned only when exactly one tool declares all of them -- an
    ambiguous shape is dropped rather than routed to the wrong tool.
    """
    if not tool_schemas:
        return None
    keys = set(_arguments_dict(data))
    if not keys:
        return None
    matches = [name for name, props in tool_schemas.items()
               if props and keys <= props]
    return matches[0] if len(matches) == 1 else None


# Conversation markers the model must never write itself. When the history
# format teaches them (``[Assistant]: ...`` / ``[Tool result for ...]``), a
# weak model copies the shape and fabricates results instead of calling.
_FABRICATED_CUT_RE = re.compile(
    r"^(?:\[Tool result for|\[System instruction\]:|\[Assistant\]:)",
    re.MULTILINE,
)


def _fence_spans(text: str) -> list:
    """``(start, end)`` spans of fenced code blocks in ``text``."""
    spans = []
    pos = 0
    while True:
        open_at = text.find("```", pos)
        if open_at == -1:
            return spans
        close_at = text.find("```", open_at + 3)
        if close_at == -1:
            spans.append((open_at, len(text)))
            return spans
        spans.append((open_at, close_at + 3))
        pos = close_at + 3


def _strip_fabricated(text: str, has_calls: bool) -> str:
    """Remove fabricated conversation markers from parsed model output.

    ``has_calls`` documents the companion rule enforced by the caller: when
    calls were accepted the tail after the last block is never appended to the
    clean text, so anything the model wrote after its block (including a
    self-written "result") cannot survive. What this function itself does,
    with or without calls:

    * a leading ``[Assistant]:`` prefix (position 0) is the history format
      leaking into the answer, not content -- strip it plus following space;
    * cut at the first line-start ``[Tool result for`` /
      ``[System instruction]:`` / ``[Assistant]:`` -- from there on the model
      is continuing the conversation by itself.

    Markers inside fenced code blocks are literal text the user asked about
    (e.g. quoting this proxy's own prompt format) and never trigger a cut.
    """
    if text.startswith("[Assistant]:"):
        text = text[len("[Assistant]:"):].lstrip()
    spans = _fence_spans(text)

    def _inside(pos: int) -> bool:
        return any(s <= pos < e for s, e in spans)

    for m in _FABRICATED_CUT_RE.finditer(text):
        if not _inside(m.start()):
            text = text[:m.start()]
            break
    return text


def _normalize_tool_name(name, allowed) -> str:
    """Resolve a model-written tool name against the declared ``allowed`` set.

    Models prefix (``functions.read``, ``default_api.read``) and vary case
    (``Read``); all of those used to be dropped as undeclared. Strip repeated
    ``functions.`` / ``default_api.`` prefixes, then adopt an allowed name
    only when exactly one candidate matches case-insensitively or by final
    dotted segment -- an ambiguous shape is still dropped rather than routed
    to the wrong tool. Returns None when nothing resolves.
    """
    if not isinstance(name, str) or not name:
        return None
    if allowed is None:
        stripped = name
        while True:
            lowered = stripped.lower()
            if lowered.startswith("functions.") or lowered.startswith("default_api."):
                stripped = stripped.split(".", 1)[1]
                if not stripped:
                    return None
            else:
                break
        return stripped or None
    if name in allowed:
        return name
    stripped = name
    while True:
        lowered = stripped.lower()
        if lowered.startswith("functions.") or lowered.startswith("default_api."):
            stripped = stripped.split(".", 1)[1]
            if not stripped:
                return None
        else:
            break
    if stripped in allowed:
        return stripped
    candidates = {a for a in allowed if a.lower() == stripped.lower()}
    if not candidates and "." in stripped:
        segment = stripped.split(".")[-1].lower()
        candidates = {a for a in allowed
                      if a.split(".")[-1].lower() == segment}
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _recover_python_payload(body):
    """Read a non-JSON fence body as a Python call or dict literal.

    Returns ``(data, reason)``: a dict payload and ``""`` on success, else
    ``(_NOTHING, reason)``. Only for fence kinds other than ``json`` -- a
    ``json`` fence that does not parse is prose, not a call attempt.
    ``read(path="x")`` yields ``{"name": "read", "arguments": {"path": "x"}}``;
    anything with positional arguments cannot be mapped to parameters and is
    dropped with its own reason.
    """
    if not isinstance(body, str):
        return _NOTHING, "body is not valid JSON"
    snippet = body.strip().strip("`").strip()
    if not snippet:
        return _NOTHING, "body is not valid JSON"
    try:
        expr = ast.parse(snippet, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        expr = None
    if isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Name):
            fname = func.id
        elif isinstance(func, ast.Attribute):
            fname = func.attr
        else:
            return _NOTHING, "body is not valid JSON"
        if expr.args:
            return _NOTHING, "python-style call with positional arguments"
        if any(kw.arg is None for kw in expr.keywords):
            return _NOTHING, "body is not valid JSON"
        try:
            kwargs = {kw.arg: ast.literal_eval(kw.value)
                      for kw in expr.keywords}
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return _NOTHING, "body is not valid JSON"
        return {"name": fname, "arguments": kwargs}, ""
    if snippet.startswith("{"):
        try:
            value = ast.literal_eval(snippet)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return _NOTHING, "body is not valid JSON"
        if isinstance(value, dict):
            return value, ""
    return _NOTHING, "body is not valid JSON"
# for it routinely, but it is validated far more strictly below: an ordinary
# JSON example in prose must never be mistaken for a real call.
_FENCE_OPEN = re.compile(r'```(tool_call|function_call|tool_code|json)'
                         r'(?![A-Za-z0-9_])[ \t]*\n?')
_FENCE_CLOSE = re.compile(r'\s*```')


def _iter_tool_blocks(text: str):
    """Yield ``(start, end, data, kind, body)`` for each tool-call-shaped block.

    The payload is located with ``JSONDecoder.raw_decode`` starting right after
    the opening fence, so a ``` that appears *inside* a string argument (the
    normal shape of a ``write`` call whose content is Markdown or code)
    terminates neither the payload nor the block. A non-greedy regex instead
    stops at the very first fence, cuts the JSON in half, and the call is lost.

    ``data`` is ``_NOTHING`` for a block that declares a tool-call fence but
    whose body is not valid JSON; it is yielded anyway so the caller can log the
    drop rather than silently leaving it in the text. ``body`` is the raw
    payload text between the fences, for the AST recovery of Python-style
    calls (``read(path="x")``) that never were JSON in the first place.

    The decoder runs non-strict: ``write``/``edit`` payloads carry the body of
    a file, and a model that pastes a multi-line string writes the newline
    itself instead of escaping it. Strict JSON rejects that, so the whole call
    would be reported as "not valid JSON" -- the tool call silently becomes
    prose. ``json.dumps`` re-escapes the string on the way out, so what is
    handed to the client is well-formed again.
    """
    dec = json.JSONDecoder(strict=False)
    pos = 0
    while True:
        m = _FENCE_OPEN.search(text, pos)
        if not m:
            return
        kind = m.group(1)
        j = m.end()
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        try:
            data, end = dec.raw_decode(text, j)
        except ValueError:
            if kind == "json":
                # Not a call attempt, just JSON prose we cannot read: skip it
                # without a log line so ordinary answers stay quiet.
                pos = m.end()
                continue
            # Bound the reported body by the closing fence so the log shows what
            # the model actually wrote.
            close = text.find("```", m.end())
            span_end = close if close != -1 else m.end()
            body = text[j:close] if close != -1 else text[j:]
            yield m.start(), (close + 3 if close != -1 else m.end()), _NOTHING, kind, body
            pos = span_end if close != -1 else m.end()
            continue
        c = _FENCE_CLOSE.match(text, end)
        body = text[j:end]
        if c:
            end = c.end()
        yield m.start(), end, data, kind, body
        pos = end


def _is_declared_call(data, allowed_names, tool_schemas) -> bool:
    """True only for a ```json block that is unmistakably a declared tool call.

    Requires a name the client actually declared *and* a recognisable argument
    payload -- either one of the ``arguments``/``args``/``input`` aliases or
    flattened parameter keys that all belong to that tool's schema. Anything
    looser would swallow JSON examples the model quotes while explaining
    something to the user.
    """
    if not isinstance(data, dict) or not allowed_names:
        return False
    name = data.get("name")
    if name not in allowed_names:
        return False
    if any(k in data for k in _ARGUMENT_KEYS):
        return True
    keys = set(data) - _CALL_META_KEYS
    props = (tool_schemas or {}).get(name)
    return bool(keys) and bool(props) and keys <= props


def parse_tool_calls(text: str, allowed_names=None, tool_schemas=None) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list, dropped).

    ``allowed_names`` optionally restricts accepted function names.
    ``tool_schemas`` maps each declared name to its parameter names; when the
    model omits ``name`` a call is recovered only if its parameters point at
    exactly one declared tool, and a flattened call keeps a meta-looking key
    (``description``, say) only when the resolved tool's schema declares it.

    ``dropped`` is a list of human-readable reasons, one per block that looked
    like a call attempt but could not be used (undeclared name, unparsable
    body, ...). Blocks that fail are left in the text so that no content is
    silently lost; the caller retries with the reasons when nothing valid
    came out of the turn.
    """
    from .gemini import log

    tool_calls = []
    dropped = []
    clean_parts = []
    last_end = 0
    for start, end, data, kind, body in _iter_tool_blocks(text):
        if kind == "json" and not _is_declared_call(data, allowed_names, tool_schemas):
            # Ordinary JSON in prose: leave it untouched and do not log it.
            continue
        if kind in ("tool_call", "function_call", "tool_code"):
            # A fence quoted mid-sentence ("e.g. ```tool_call ...") or pasted
            # from a file is an explanation, not a call. Only fences that
            # start their line (modulo whitespace) execute.
            line_start = text.rfind("\n", 0, start) + 1
            if text[line_start:start].strip():
                continue
        reason = ""
        if data is _NOTHING and kind != "json":
            # A fence that declares a call but holds no JSON: maybe a
            # Python-style call (``read(path="x")``) or a Python dict.
            data, reason = _recover_python_payload(body)
        parsed = None
        if isinstance(data, dict):
            name = data.get("name")
            if not name:
                name = _infer_tool_name(data, tool_schemas)
                if name:
                    log(f"Inferred missing tool name '{name}' from its parameters")
            if not name:
                reason = "no tool name"
            else:
                resolved = _normalize_tool_name(name, allowed_names)
                if allowed_names is not None and resolved is None:
                    valid = ", ".join(sorted(allowed_names))
                    reason = f"undeclared tool '{name}' (valid: {valid})"
                else:
                    if resolved is not None:
                        name = resolved
                    props = (tool_schemas or {}).get(name)
                    args, degraded = _arguments_json(
                        extract_arguments(data, props))
                    if degraded:
                        log(f"tool_call '{name}': arguments unusable ({degraded}), "
                            "sending an empty object so the client can report it")
                    if _is_placeholder_args(args):
                        reason = (f"tool_call '{name}' copied the EXAMPLE_ "
                                  "placeholder from the prompt -- fill in real values")
                    else:
                        parsed = {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
        elif not reason:
            reason = ("body is not valid JSON" if data is _NOTHING
                      else "unrecognised payload")
        if parsed is None:
            dropped.append(reason or "unrecognised payload")
            log(f"Dropping tool_call block: {reason}: {text[start:end][:300]}")
            continue
        clean_parts.append(text[last_end:start])
        last_end = end
        tool_calls.append(parsed)
    if tool_calls:
        # The two-state rule: a turn either calls or answers. Anything after
        # the last block -- prose, and in particular a self-written
        # "[Tool result ...]" -- is never part of the answer.
        clean = "".join(clean_parts)
    else:
        clean_parts.append(text[last_end:])
        clean = "".join(clean_parts)
    clean = _strip_fabricated(clean, bool(tool_calls)).strip()
    return clean, tool_calls, dropped


def looks_like_tool_call(text: str) -> bool:
    """Whether ``text`` holds a block ``parse_tool_calls`` would actually emit.

    One response carries its answer and its tool call as separate texts, so a
    caller that picks between them by length can let a summary win over a call
    the model really made -- the client then receives prose claiming a tool
    was used and no call to run. This is the tie-breaker for that choice.

    Only the canonical shape counts. A fenced JSON example that merely happens
    to carry a top-level ``name`` is not a call, so preferring candidates that
    pass here can never swap the answer for a snippet quoted in prose: anything
    rejected simply stays out of the running and the caller falls back to
    comparing lengths as before.
    """
    for _start, _end, data, kind, _body in _iter_tool_blocks(text):
        if not isinstance(data, dict) or not data.get("name"):
            continue
        # ``{"name": ..., "arguments": ...}`` is a call under any fence; a
        # bare ``json`` fence needs the argument wrapper, because ``{"name":
        # "Alice"}`` in an example is exactly the prose this must not catch.
        if kind != "json" or any(k in data for k in _ARGUMENT_KEYS):
            return True
    return False


# Gemini's own generation step failing produces this canned sentence rather
# than an answer. It reaches the client as a normal 200, so the caller reads it
# as "the model replied" and stops -- when in fact nothing was generated and a
# single retry almost always succeeds.
#
# Both detectors below match against ``_fold_diacritics`` output, so their
# patterns are written in plain ASCII and recognise Vietnamese typed with or
# without accents.
_UPSTREAM_ERROR_RE = re.compile(
    r"^\s*(?:i encountered an error doing what you asked\.?\s*"
    r"could you try again\?*"
    r"|sorry, something went wrong\.?"
    r"|da co loi xay ra\.?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Phrases that mean "I am about to act" or "I cannot reach that" --
# i.e. the model announced an action it never performed because it never
# emitted the tool call. Covers both read verbs (read/check/look/...) and
# implement verbs (write/create/update/run/...) in English and Vietnamese:
# without the latter an agent stops after one step ("I'll create the file"
# with no block reads as a finished answer).
_MISSED_TOOL_RE = re.compile(
    r"(khong\s+the\s+truy\s+cap|khong\s+co\s+quyen|khong\s+duoc\s+truy\s+cap"
    r"|khong\s+tim\s+thay|khong\s+the\s+doc"
    r"|cannot\s+access|can'?t\s+access"
    r"|don'?t\s+have\s+access|do\s+not\s+have\s+access"
    r"|no\s+(?:file\s+|tool\s+)?access"
    r"|not\s+(?:permitted|allowed)\s+to\s+read"
    r"|de\s+minh\s+(?:doc|kiem\s+tra|viet|tao|sua|chay|cap\s+nhat|thuc\s+hien|xem|mo)"
    r"|minh\s+se\s+(?:doc|viet|tao|sua|chay|cap\s+nhat|thuc\s+hien|kiem\s+tra)"
    r"|toi\s+se\s+(?:doc|viet|tao|sua|chay|cap\s+nhat|thuc\s+hien|kiem\s+tra)"
    r"|let'?s\s+(?:read|check|look|see|examine|inspect|write|create|update|edit|run|implement|fix|build)"
    r"|let\s+us\s+(?:read|check|look|see|examine|inspect|write|create|update|edit|run|implement|fix|build)"
    r"|let\s+me\s+(?:read|check|look|see|examine|inspect|write|create|update|edit|run|execute|implement|fix|build|apply|test)"
    r"|i(?:'ll|\s+will)\s+(?:read|check|look|see|examine|inspect|write|create|update|edit|modify|fix|run|execute|implement|build|apply|test|continue))",
    re.IGNORECASE,
)

# Longer guard for verbose models that explain before emitting tool_call.
# A long plan summary (>600 chars) followed by a promise to act is still a
# stalled turn, not a real answer.
_MAX_MISSED_TOOL_CHARS = 1200


def _fold_diacritics(text: str) -> str:
    """Strip accents so patterns match Vietnamese typed with or without them.

    ``đ``/``Đ`` have no canonical decomposition, so they are folded explicitly.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("đ", "d").replace("Đ", "D")


def is_required_tool_choice(tool_choice) -> bool:
    """True when the client demands a tool call rather than allowing one.

    Chat Completions nests the target under ``function``; the Responses API
    drops that nesting and names it directly, so both shapes have to count --
    recognising only the nested one silently disabled the required-tool path
    for every Responses client.
    """
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        return bool(tool_choice.get("function") or tool_choice.get("name"))
    return False


def looks_like_upstream_error(text: str) -> bool:
    """True when the response is the upstream's canned failure, not an answer."""
    if not text:
        return False
    return bool(_UPSTREAM_ERROR_RE.match(_fold_diacritics(text)))


def looks_like_missed_tool_call(text: str) -> bool:
    """True when a response *intended* to use a tool but never emitted a block.

    All three conditions must hold: the text announces or refuses an action,
    it is short enough to be a stalled first turn rather than a real answer,
    and it carries no tool fence (a fence means the model did try). Returning
    True costs one extra upstream call; returning True for a genuine answer
    would make every short reply do double duty, hence the length guard.
    """
    if not text or len(text) >= _MAX_MISSED_TOOL_CHARS:
        return False
    if "```tool_call" in text or "```function_call" in text:
        return False
    folded = _fold_diacritics(text)
    if folded.rstrip().endswith(":"):
        return True
    return bool(_MISSED_TOOL_RE.search(folded))


# Phrases where the model hands the work back to the user instead of acting:
# a how-to for code it already read. A verbatim bug quote (the prompt's own
# "quote the exact lines" rule) has proposed code too, so the handoff phrase
# is required -- a quote that explains a bug never tells the user to apply it.
_DESCRIBED_EDIT_RE = re.compile(
    r"\b(?:you\s+can|you\s+need\s+to|you\s+should|here\s+is\s+how\s+you\s+can|"
    r"follow\s+these\s+steps|update\s+\S+\s+as\s+follows|make\s+the\s+"
    r"following\s+changes?|to\s+do\s+this,?|run\s+\S+\s+to\s+verify|"
    r"since\s+tool\s+calls?\s+are\s+no\s+longer\s+available|"
    r"unable\s+to\s+execute\s+tool|i\s+would\s+be\s+happy\s+to\s+help\s+you\s+"
    r"(?:identify|implement|fix|update))\b",
    re.IGNORECASE,
)
_DIFF_MARKER_RE = re.compile(
    r"(?:^|\n)(?:\+\+\+|---|@@\s)", re.MULTILINE,
)


def looks_like_described_edit(text: str) -> bool:
    """True when the reply describes an edit instead of emitting the call.

    Both halves must hold: proposed code (a diff hunk or a fenced code block)
    AND a handoff phrase telling the user to apply it. Either half alone is
    legitimate -- a quoted bug snippet, or prose guidance with no code.
    """
    if not text or "```tool_call" in text or "```function_call" in text:
        return False
    has_code = bool(_DIFF_MARKER_RE.search(text)
                    or re.search(r"```(?:python|javascript|js|json|diff|bash|"
                                 r"typescript|ts)\b", text, re.IGNORECASE))
    if not has_code:
        return False
    return bool(_DESCRIBED_EDIT_RE.search(_fold_diacritics(text)))


# File-shaped tokens a sentence can point at: `README.md`, `server.py`,
# `src/server.py`, `gemini-web2api.log`. The optional ``(dir/)*`` prefix
# keeps project paths intact so callers can detect the project root; without
# it a "detect root via f.split('/')" block can never fire.
_FILE_REF_RE = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*[\w][\w.-]*\.(?:"
    r"md|py|pyi|js|jsx|ts|tsx|json|jsonc|txt|ya?ml|toml|cfg|conf|ini|env|"
    r"java|kt|c|cc|cpp|h|hpp|go|rs|rb|php|html|css|scss|less|sql|sh|bash|ps1|"
    r"xml|csv|lock|log|rst|bat|cmd|vue|svelte|proto|gradle|properties"
    r")(?![\w-])",
    re.IGNORECASE,
)

# Bare `base.ext` tokens with these extensions are far more often member
# access (`self.cmd`, `process.env`, `math.log(`, `docs.rs`, `args.go`) than
# real files. They only count when path-like (`a/b.log`) or quoted/backticked.
_RISKY_BARE_EXTS = frozenset({"cmd", "env", "log", "rs", "go"})

# Names that end in a source extension but are runtimes, frameworks or
# libraries rather than files. `Node.js` and friends would otherwise look like
# an unread file for every turn that mentions them -- no tool call ever
# "opens" them, so the request would be retried until the conversation ends.
_NON_FILE_NAMES = frozenset({
    "node.js", "node.mjs", "node.cjs", "next.js", "nuxt.js", "vue.js",
    "react.js", "react-dom.js", "express.js", "jquery.js", "d3.js",
    "three.js", "ember.js", "backbone.js", "socket.js", "deno.js",
})

def _canonical_args(arguments) -> str:
    """Normalise a call's arguments so repeats compare equal.

    Argument key order is a serialisation accident, not a different call:
    non-strict ``json.loads`` plus ``sort_keys`` dump. Anything that is not
    JSON falls back to the stripped raw text.
    """
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments, strict=False)
        except (json.JSONDecodeError, ValueError):
            return arguments.strip()
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(arguments)


def _repeatable_calls(messages) -> set:
    """``{(name, canonical_args)}`` this conversation already has results for.

    Every assistant ``tool_calls`` entry paired with a later ``role: "tool"``
    result carrying the same ``tool_call_id``. The retry loop uses this to
    recognise a model that keeps emitting a call it already got the answer
    to, instead of executing the same call again.

    A mutating tool (edit/write/bash/...) changes the world the earlier
    results described: after one completes, every earlier entry is dropped.
    Re-running the same ``pytest`` or re-reading a just-edited file is then
    progress, not a loop.
    """
    calls_by_id = {}
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            tcid = tc.get("id")
            if tcid and isinstance(fn, dict):
                calls_by_id[tcid] = (fn.get("name"),
                                     _canonical_args(fn.get("arguments")))
    done = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            continue
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id", "")
        key = calls_by_id.get(tcid)
        name = key[0] if key else (msg.get("name") or "")
        if key is not None:
            done.add(key)
        if isinstance(name, str) and _MUTATING_TOOL_RE.search(name):
            done.clear()
    return done


def _google_repeatable(contents) -> set:
    """Chat-path :func:`_repeatable_calls` for Google native ``contents``.

    Model parts carrying ``functionCall`` are paired with later
    ``functionResponse`` parts by name (falling back to oldest-first when the
    name matches nothing): each response completes one outstanding call, and
    completed calls are what the retry loop must not emit again. A completed
    mutating call clears earlier entries: state changed, so repeats are new
    work.
    """
    pending = []
    done = set()
    for content in contents or []:
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("functionCall"), dict):
                fc = part["functionCall"]
                pending.append((fc.get("name"),
                                _canonical_args(fc.get("args"))))
            elif isinstance(part.get("functionResponse"), dict):
                fr = part["functionResponse"]
                idx = next((i for i, (pending_name, _a) in enumerate(pending)
                            if pending_name == fr.get("name")), None)
                if idx is None and pending:
                    idx = 0
                if idx is not None:
                    key = pending.pop(idx)
                    done.add(key)
                    if isinstance(key[0], str) and _MUTATING_TOOL_RE.search(key[0]):
                        done.clear()
                        # Keep the mutating call itself out: re-emitting the
                        # exact edit is a loop, re-reading after it is not.
    return done


# Tool names that change state (files, processes, environment). A repeatable
# entry from before one of these completed is stale: the world changed.
_MUTATING_TOOL_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:write|edit|create|update|delete|remove|mkdir|rmdir|"
    r"bash|shell|exec|run|apply_patch|apply|patch|append|save|move|copy|"
    r"touch|chmod|truncate)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)


# "dùng tool", "đọc file", "review source", "use the tool", ... -- matched
# against the folded text with whitespace runs collapsed to single spaces
# (never deleted: deleting them is what glued "because"+"tool" into a phrase
# containing "use"+"tool"). Every alternative starts at a word boundary with
# real whitespace between components, so intra-word lookalikes ("because
# tool", "demo file", "thread files", "review file") stay silent while the
# real demands still fire.
_TOOL_DEMAND_RE = re.compile(
    r"\b(?:su\s+|hay\s+|phai\s+|van\s+phai\s+|bat\s+buoc\s+|co\s+)?dung\s+"
    r"(?:la\s+)?(?:tool|tools|cong\s+cu|cac\s+cong\s+cu)"
    r"|\b(?:doc|kiem\s+tra|xem|mo)(?:\s+rai)?\s+"
    r"(?:file|files|tep|thu\s+muc|cac\s+file|cac\s+tep)"
    r"|\b(?:review|kiem\s+tra|doc)(?:\s+la)?\s+"
    r"(?:source|code|ma\s+nguon|cac\s+file)"
    r"|\b(?:use|call|run|invoke)\s+(?:the\s+)?(?:tool|tools)"
    r"|\b(?:read|open|view|check|inspect)\s+(?:the\s+)?(?:file|files)",
    re.IGNORECASE,
)

# A request to review/audit/inspect source code (matched against folded,
# whitespace-collapsed text, like _TOOL_DEMAND_RE). Kept separate because a
# listing-only turn (glob/list) must still count as owed here, while every
# other demand keeps the old "any call ends the check" rule.
#
# Verbs are split: strong verbs (review/audit/evaluate/...) combine with any
# noun, including plan/document/design/spec. Weak verbs (check/find/read/...)
# only combine with code nouns -- "read the plan then implement" is an
# implement turn, not a source review, and must not be forced to read modules.
#
# The broad noun set also covers the system under review when the user names
# it instead of the code ("review how this proxy loads the session cookie"):
# with a strong verb the request is unambiguously a codebase inspection, so
# proxy/server/service/... count. Weak verbs keep the narrow code nouns, so
# "check the server status" stays silent.
_SOURCE_REVIEW_VERB_RE = re.compile(
    r"\b(?:review|audit|inspect|examine|check|find|kiem\s+tra|soat|tim|xem\s+xet|doc|danh\s+gia|danhgia)\b",
    re.IGNORECASE,
)
_SOURCE_STRONG_VERB_RE = re.compile(
    r"\b(?:review|audit|inspect|examine|soat|danh\s+gia|danhgia|evaluate|assess)\b",
    re.IGNORECASE,
)
_SOURCE_CODE_NOUN_RE = re.compile(
    r"\b(?:source|code|codebase|ma\s+nguon|implementation|repository|repo)\b",
    re.IGNORECASE,
)
_SOURCE_NOUN_RE = re.compile(
    r"\b(?:source|code|codebase|ma\s+nguon|implementation|repository|repo|ke\s+hoach|kehoach|plan|document|tai\s+lieu|tai-lieu|thiet\s+ke|thietke|design|specification|spec|proxy|server|service|app|module|package|project)\b",
    re.IGNORECASE,
)


def _is_source_review(folded: str) -> bool:
    """True when folded text asks for a review of source/code."""
    if _SOURCE_STRONG_VERB_RE.search(folded) and _SOURCE_NOUN_RE.search(folded):
        return True
    return bool(_SOURCE_REVIEW_VERB_RE.search(folded)
                and _SOURCE_CODE_NOUN_RE.search(folded))


def _is_file_review(folded: str, scannable: str) -> bool:
    """True when folded text aims a review verb at a named file.

    "Tiep tuc review server.py di." carries no source noun, so
    :func:`_is_source_review` misses it -- yet answering it from a glob
    listing alone is the same fabrication. A review verb plus a file-shaped
    token counts as a review of that file ("review file" with no file named
    stays silent; runtime names like ``Node.js`` never count as files).
    """
    if not _SOURCE_REVIEW_VERB_RE.search(folded):
        return False
    return any(f for f in _files_in(scannable)
               if f.lower() not in _NON_FILE_NAMES)


# Tool names that read content (not just list names). A grep/search with no
# path in its args still read something, so the name alone ends the check.
_READER_TOOL_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:read|cat|view|open|show|grep|ripgrep|search|head|tail|less|sed|fetch)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

# A glob-only turn answered from filenames alone is the failure this guards:
# the model listed names but never opened one.
SOURCE_REVIEW_OWED = ("source review requested but no file content was read "
                      "(only listed)")


def is_read_retry_reason(reason) -> bool:
    """True when a retry reason means "no file content was read yet"."""
    return bool(reason) and ("never read" in reason
                             or reason == SOURCE_REVIEW_OWED)


def _last_user_index(messages) -> int:
    """Index of the last ``role == "user"`` dict, else -1."""
    idx = -1
    for i, msg in enumerate(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            idx = i
    return idx


def _turn_tool_calls(messages, start_idx) -> list:
    """Every ``tool_calls`` dict in messages after ``start_idx``."""
    calls = []
    for msg in (messages or [])[start_idx + 1:]:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                calls.append(tc)
    return calls


def _tool_arguments_dict(fn) -> dict:
    """Parse a ``tool_calls`` function payload's arguments to a dict or {}."""
    args = (fn or {}).get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args, strict=False)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# Argument keys that carry a file path. Only these are scanned for evidence:
# scanning the whole serialised args is what counted `glob
# {"pattern": "**/hand3d.py"}` as "read hand3d.py" and `bash ls x.py` as a
# read. `pattern`/`command`/`query` deliberately excluded: a listing pattern
# or a shell one-liner is not content.
_PATH_ARG_KEYS = frozenset({
    "path", "filepath", "file_path", "file", "filename", "file_name",
    "target", "target_file", "dirpath", "directory", "folder",
})


def _is_reader_tool(name) -> bool:
    return isinstance(name, str) and bool(_READER_TOOL_RE.search(name))


def _turn_result_ids(messages, start_idx) -> set:
    """``tool_call_id`` values with a ``role == "tool"`` result this turn."""
    ids = set()
    for msg in (messages or [])[start_idx + 1:]:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if tcid:
                ids.add(tcid)
    return ids


def files_read_in_turn(messages) -> set:
    """Lowercased file names whose content this turn's tool calls opened.

    Only reader tools (``read``/``cat``/``view``/``grep``/...) count, only
    path-like argument keys are scanned, and a call without a paired tool
    result does not count: a ``glob`` pattern or an unanswered call is not
    evidence the model saw content.
    """
    start_idx = _last_user_index(messages)
    result_ids = _turn_result_ids(messages, start_idx)
    files = set()
    for tc in _turn_tool_calls(messages, start_idx):
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if not _is_reader_tool(name):
            continue
        if tc.get("id") and result_ids and tc.get("id") not in result_ids:
            continue
        args = _tool_arguments_dict(fn)
        if args:
            vals = [v for k, v in args.items()
                    if k in _PATH_ARG_KEYS and isinstance(v, str)]
            if vals:
                files |= {f.lower() for f in _files_in(*vals)}
                continue
            # Reader with no path key (grep query only): content was still
            # read, but no file can be named -- handled by _reads_this_turn.
            continue
        raw = fn.get("arguments")
        if isinstance(raw, str):
            found = {f.lower() for f in _files_in(raw)}
            if found:
                files |= found
    return files


def _reads_this_turn(messages) -> int:
    """Number of distinct content reads in this turn."""
    files = files_read_in_turn(messages)
    if files:
        return len(files)
    start_idx = _last_user_index(messages)
    result_ids = _turn_result_ids(messages, start_idx)
    for tc in _turn_tool_calls(messages, start_idx):
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if _is_reader_tool(name):
            if tc.get("id") and result_ids and tc.get("id") not in result_ids:
                continue
            return 1
    return 0


def _source_reads_this_turn(messages) -> int:
    """Number of distinct SOURCE CODE (.py) reads in this turn."""
    files = files_read_in_turn(messages)
    return len({f for f in files if f.endswith(".py")})


def review_demanded(messages) -> bool:
    """True when the newest user turn asks for a source/code review."""
    last_user = None
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = _stringify_content(msg.get("content", ""))
    if not last_user:
        return False
    scannable = _strip_code_spans(last_user)
    folded = _fold_diacritics(scannable)
    folded = re.sub(r"\s+", " ", folded).strip()
    return bool(_is_source_review(folded)
                or _is_file_review(folded, scannable))


def declared_unread(messages) -> list:
    """Files the model named in this turn's prose but never read, else ``[]``.

    Ordered and unique: file-shaped tokens in assistant ``content`` since the
    last user turn (code spans blanked first, so pasted logs/code do not
    count; runtime names like ``Node.js`` never count), minus the files this
    turn's tool calls opened (compared case-insensitively).
    """
    start_idx = _last_user_index(messages)
    read_lower = files_read_in_turn(messages)
    seen = []
    for msg in (messages or [])[start_idx + 1:]:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for f in _files_in(_strip_code_spans(
                _stringify_content(msg.get("content", "")))):
            if f.lower() in _NON_FILE_NAMES or f.lower() in read_lower:
                continue
            if f not in seen:
                seen.append(f)
    return sorted(seen)


def fresh_declared_unread(text, messages) -> list:
    """Like :func:`declared_unread` but for one fresh reply string.

    The retry loop calls this on the reply it just received, before that
    reply is appended to ``messages``.
    """
    if not isinstance(text, str) or not text:
        return []
    read_lower = files_read_in_turn(messages)
    seen = []
    for f in _files_in(_strip_code_spans(text)):
        if f.lower() in _NON_FILE_NAMES or f.lower() in read_lower:
            continue
        if f not in seen:
            seen.append(f)
    return sorted(seen)


def listed_turn_files(messages) -> list:
    """File names from ``role == "tool"`` contents since the last user turn.

    Ordered and unique: the retry nudge names these so the model reads what
    it just listed instead of guessing. Bodies longer than ~2 KB (a just-read
    file's content) are skipped: scanning a full file body is what promoted
    incidental mentions (`config.json` quoted in code) into "listed files"
    and pushed ``required_reads`` to 2 on turns that listed nothing. Short
    listing outputs (glob/ls results) are unaffected.
    """
    start_idx = _last_user_index(messages)
    seen = []
    for msg in (messages or [])[start_idx + 1:]:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _stringify_content(msg.get("content", ""))
        if len(body) > 2000:
            continue
        for f in _files_in(body):
            if f.lower() in _NON_FILE_NAMES:
                continue
            if f not in seen:
                seen.append(f)
    return seen


def _strip_code_spans(text: str) -> str:
    """Remove fenced and inline code from ``text`` for demand/file scanning.

    A pasted log or code block can name files (``server.py``) and use tool
    words ("read the file" inside a comment) without the user asking for
    anything -- scanning the raw text turns every paste into a phantom "never
    read" retry. Paired fences are blanked; an unpaired opening fence blanks
    everything to the end (the paste was cut off, not prose).
    """
    if not isinstance(text, str) or not text:
        return ""
    blanked = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    blanked = re.sub(r"```.*$", " ", blanked, flags=re.DOTALL)
    blanked = re.sub(r"`[^`\n]+`", " ", blanked)
    return blanked


def _files_in(*chunks) -> set:
    """File-shaped names mentioned anywhere in ``chunks`` (strings or JSON).

    Filters member access (`self.cmd`, `process.env`) and call expressions
    (`math.log(`): a token followed by ``(`` is code, not a file. Bare
    tokens with high-collision extensions (cmd/env/log/rs/go) need a path
    separator or quoting to count.
    """
    blob = " ".join(c for c in chunks if isinstance(c, str))
    out = set()
    for m in _FILE_REF_RE.finditer(blob):
        token = m.group(0)
        end = m.end()
        j = end
        while j < len(blob) and blob[j] in " \t":
            j += 1
        if j < len(blob) and blob[j] == "(":
            continue
        if "/" not in token and "\\" not in token:
            ext = token.rsplit(".", 1)[-1].lower() if "." in token else ""
            if ext in _RISKY_BARE_EXTS:
                start = m.start()
                before = blob[start - 1] if start > 0 else ""
                after = blob[end] if end < len(blob) else ""
                if before not in ("`", "'", '"', "/", "\\") \
                        and after not in ("`", "'", '"'):
                    continue
        out.add(token)
    return out


def _conversation_called_files(messages) -> set:
    """Every file name a *reader* tool call in this conversation opened.

    Listing/writing tools do not count: a ``glob`` pattern that mentions a
    file is not evidence the model read it, and without this the `uncovered`
    check is disabled by the very discovery step it prescribes.
    """
    called = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not _is_reader_tool(fn.get("name") or ""):
                continue
            args = _tool_arguments_dict(fn)
            if args:
                vals = [v for k, v in args.items()
                        if k in _PATH_ARG_KEYS and isinstance(v, str)]
                if vals:
                    called |= _files_in(*vals)
                    continue
            raw = fn.get("arguments")
            if isinstance(raw, str):
                called |= _files_in(raw)
            elif args:
                called |= _files_in(json.dumps(args, ensure_ascii=False))
    return called


# "create/write/generate/make a file ..." (EN + VI): files named in such a
# turn are to be *created*, not read. The `uncovered` branch must not demand
# reading a file that does not exist yet -- the retry nudge would otherwise
# tell the model to "read" it and the read fails.
_FILE_CREATE_RE = re.compile(
    r"\b(?:create|write|generate|make|add|tao|tao\s+moi|viet|tao\s+file|"
    r"tao\s+tep|ghi\s+file|thiet\s+ke\s+moi)\b",
    re.IGNORECASE,
)


def _created_files(scannable: str, folded: str) -> set:
    """File tokens exempt from the `uncovered` check (creation intent)."""
    if not _FILE_CREATE_RE.search(folded):
        return set()
    return {f.lower() for f in _files_in(scannable)}


# A request to CHANGE the code (fix/implement/update/...), matched against
# folded text like the review heuristics. This is the implement-side twin of
# `_is_source_review`: the failure it guards is the survey-stall, where the
# model globs/greps to locate the code and then answers from the snippets --
# describing the edit instead of reading the file and making it.
#
# A bare change verb is not enough ("double it" in a maths chat): it must aim
# at code -- a file token, a code/config noun, or an explicit act phrase
# ("make it so"). Creation verbs (create/write/generate) are excluded: a file
# that does not exist yet must not be demanded for reading.
_CHANGE_VERB_RE = re.compile(
    r"\b(?:fix|repair|implement|add|introduce|update|upgrade|change|modify|"
    r"refactor|remove|delete|drop|double|halve|increase|decrease|migrate|"
    r"sua|them|cap\s+nhat)\b",
    re.IGNORECASE,
)
_CHANGE_NOUN_RE = re.compile(
    r"\b(?:source|code\w*|implementation|repository|repo\w*|proxy|server|"
    r"service|app|module\w*|package|project|config\w*|setting\w*|option\w*|"
    r"endpoint\w*|rout\w*|test\w*|retr\w*|auth\w*|health\w*|bug\w*|"
    r"securit\w*|vulnerab\w*)\b",
    re.IGNORECASE,
)
_CHANGE_ACT_RE = re.compile(
    r"\b(?:make\s+it\s+so|make\s+it\s+work|get\s+it\s+done|do\s+it|"
    r"go\s+ahead|cu\s+lam|lam\s+di)\b",
    re.IGNORECASE,
)

# Tool names that act on the world (not just observe it). A change turn that
# emitted one of these has done the work; a turn with only glob/grep/read
# has surveyed but not acted.
_ACT_TOOL_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:edit|write|create|apply|patch|bash|shell|exec|run|"
    r"pytest|npm|make)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)


def _is_change_request(folded: str, scannable: str) -> bool:
    """True when folded text asks for a code change (fix/implement/update)."""
    if _CHANGE_ACT_RE.search(folded):
        return True
    if not _CHANGE_VERB_RE.search(folded):
        return False
    if _CHANGE_NOUN_RE.search(folded):
        return True
    return any(f for f in _files_in(scannable)
               if f.lower() not in _NON_FILE_NAMES)


def _turn_act_calls(messages, start_idx) -> bool:
    """True when this turn already emitted an acting (edit/write/run) call."""
    for tc in _turn_tool_calls(messages, start_idx):
        fn = tc.get("function") or {}
        if isinstance(fn, dict) and _ACT_TOOL_RE.search(fn.get("name") or ""):
            return True
    return False


def change_demanded(messages) -> bool:
    """True when the newest user turn asks for a code change."""
    last_user = None
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = _stringify_content(msg.get("content", ""))
    if not last_user:
        return False
    scannable = _strip_code_spans(last_user)
    folded = _fold_diacritics(scannable)
    folded = re.sub(r"\s+", " ", folded).strip()
    return _is_change_request(folded, scannable)


def pending_tool_request(messages) -> str:
    """Why the newest user turn still owes the model a tool call, else ``None``.

    This is the check that catches a *fabricated* answer -- the failure mode
    where the user asks about ``README.md`` and the model replies as if it had
    opened it, without ever emitting a block. The reply contains no fence, no
    refusal and nothing grammatically wrong, so the response-side heuristic
    above never fires; the only thing provably false is that no tool call in
    the whole conversation has touched the file being discussed.

    Two independent shapes are reported: a file the newest user turn names
    that nothing has read, and a turn that demands tool use without one
    (covering requests with no file to name -- "review source code", say).

    Two guards keep this from burning a call on every finished answer:

    * a tool call made *this turn* ends the search. The text that follows is
      built on a real result, even when it names a file the call did not spell
      out, and replacing a completed summary with the retry's output is how a
      valid answer turns into a tool call nobody asked for;
    * names are compared case-insensitively, because ``readme.md`` typed by
      the user and ``README.md`` passed to ``read`` are the same file -- and
      on Windows they usually are literally.

    Opencode treats such a reply as the final answer and stops, so one turn of
    this ends the session with an answer invented from the filename alone.
    """
    if not isinstance(messages, list):
        return None

    last_user_idx = -1
    last_user = None
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i
            last_user = _stringify_content(msg.get("content", ""))

    if not last_user:
        return None

    # Pasted logs and code name files the user never asked about; scan what
    # remains after blanking them, for both the file check and the demand
    # match below.
    scannable = _strip_code_spans(last_user)
    folded = _fold_diacritics(scannable)
    folded = re.sub(r"\s+", " ", folded).strip()
    # A listing-only turn (glob/list) on a source-review request must stay
    # owed: filenames alone are not content, and answering from them is the
    # hallucinated review this check exists to prevent.
    is_generic_source_review = _is_source_review(folded)
    is_file_review = _is_file_review(folded, scannable)
    source_review = is_generic_source_review or is_file_review
    turn_has_call = any(isinstance(msg, dict) and msg.get("tool_calls")
                        for msg in messages[last_user_idx + 1:])
    required_reads = 1
    if (is_generic_source_review and len(listed_turn_files(messages)) >= 2):
        # Require reading most of the listed files for a genuine source review.
        # Not just 2 -- model would read minimum and hallucinate the rest.
        # File reviews (review a specific file) are handled by the `uncovered` branch.
        # Generic source reviews (review source/code/plan) require reading source code.
        listed = listed_turn_files(messages)
        # Detect project root from listed files (first path component).
        # Count Python files in each project root, use the one with most files.
        from collections import Counter
        project_counts = Counter()
        for f in listed:
            if f.endswith(".py") and not f.startswith("__pycache__/"):
                parts = f.split("/")
                if len(parts) > 1:
                    project_counts[parts[0]] += 1
        if project_counts:
            main_project = project_counts.most_common(1)[0][0]
            core_files = [f for f in listed if f.startswith(main_project + "/") and f.endswith(".py")]
            if len(core_files) >= 5:
                required_reads = max(5, len(core_files) - 1)
            else:
                required_reads = 2
        else:
            required_reads = 2
    reads = _reads_this_turn(messages)
    source_reads = _source_reads_this_turn(messages)
    called_lower = {f.lower() for f in _conversation_called_files(messages)}
    # An implement/change request is owed until the turn shows real contact
    # with the code: an acting call (edit/write/run), a path read, or a read
    # from an earlier turn. A glob/grep-only survey is not contact -- the
    # snippets are leads, not evidence -- so the early "a call ends the
    # search" return must not swallow it.
    change_owed = (
        _is_change_request(folded, scannable)
        and not _turn_act_calls(messages, last_user_idx)
        and not files_read_in_turn(messages)
        and not called_lower
    )
    if turn_has_call and (reads >= required_reads or not source_review) \
            and not change_owed:
        return None

    created = _created_files(scannable, folded)
    uncovered = sorted(
        f for f in _files_in(scannable)
        if f.lower() not in called_lower and f.lower() not in _NON_FILE_NAMES
        and f.lower() not in created
    )
    if uncovered:
        return "requested file(s) never read: " + ", ".join(uncovered)

    if change_owed:
        return "change requested but no file content was read or edited"

    # Only generic source reviews ("review source") enforce reading every
    # file the model names in its prose. Named-file reviews are handled by
    # the `uncovered` branch above (user-named files from the latest message).
    declared = declared_unread(messages)
    if is_generic_source_review and declared and turn_has_call:
        return "declared file(s) never read: " + ", ".join(declared)

    # For generic source reviews, require reading source code files (.py).
    # Use source_reads (only .py files) instead of total reads.
    effective_reads = source_reads if is_generic_source_review else reads
    if source_review and turn_has_call and effective_reads < required_reads:
        return SOURCE_REVIEW_OWED

    if _TOOL_DEMAND_RE.search(folded):
        return "user demanded tool use and none happened"
    return None


def build_response_format_instruction(response_format) -> str:
    """Build a prompt instruction from an OpenAI ``response_format`` value."""
    if not isinstance(response_format, dict):
        return ""
    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return (
            "\n\nIMPORTANT: Respond with a single valid JSON object only. "
            "Do not include prose, explanations or Markdown code fences."
        )
    if rf_type == "json_schema":
        schema = response_format.get("json_schema", {})
        if isinstance(schema, dict) and "schema" in schema:
            schema = schema["schema"]
        return (
            "\n\nIMPORTANT: Respond with a single valid JSON value that strictly "
            "conforms to the following JSON Schema. Do not include prose, "
            "explanations or Markdown code fences.\n"
            f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )
    return ""


def strip_code_fence(text: str) -> str:
    """Remove a single wrapping Markdown code fence (e.g. ```json ... ```)."""
    if not text:
        return text
    stripped = text.strip()
    m = re.match(r'^```[a-zA-Z0-9_-]*[ \t]*\n?(.*?)\n?```$', stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


# ─── Google Native API helpers ─────────────────────────────────────────────────


def build_tool_prompt(tool_defs: list) -> str:
    """Build natural tool-use prompt for Gemini Web that avoids prompt-injection detection."""
    tool_spec = json.dumps(tool_defs, indent=2, ensure_ascii=False)
    return (
        "# Tool Use\n\n"
        "You can call the following tools to help accomplish tasks. "
        "These tools connect to the user's local environment and will execute when called.\n\n"
        "Call format (use this exact format):\n"
        "```function_call\n"
        '{"name": "<tool_name>", "args": {<arguments>}}\n'
        "```\n\n"
        "When calling tools:\n"
        "- Each turn do exactly ONE of: (A) you need to read or change "
        "something -> output ONLY function_call block(s); (B) you already have "
        "everything -> answer as plain text with NO block. Never both, and "
        "never any prose after the block.\n"
        "- Before answering a request to review, audit, inspect or find bugs in "
        "source code, you MUST read the actual contents of the relevant files with "
        "a read/search tool first -- reading one file is not reviewing a codebase. "
        "Listing or globbing file names is not reading.\n"
        "- Output ONLY the function_call block(s), nothing else\n"
        "- You may call multiple tools with multiple blocks\n"
        "- Fixed flow for EVERY turn: do you already have a [Tool result "
        "for ...] proving everything you need? If YES -> answer as plain "
        "text. If NO -> output ONLY function_call block(s) and stop; a "
        "guess or instructions for the user are never an answer.\n"
        "- Every parameter belongs inside the \"args\" object, keyed exactly "
        "like the parameters properties -- never beside \"name\".\n"
        "- Never state that you have read a file, run a command or inspected "
        "the code unless a [Tool result for ...] message proves it.\n"
        "- Never claim the tools are unavailable or that you lack permission: "
        "they are connected and will run.\n"
        "- Never write [Tool result for ...] yourself, and never invent a "
        "result: emit the block and stop.\n"
        "- Search results are leads, not evidence: read matched files "
        "before answering or editing.\n"
        "- After editing, re-read the changed lines before reporting "
        "success.\n"
        "- Triple backticks inside a string argument (Markdown or code the user "
        "asked you to save) are literal text: keep all three of them. A fence "
        "inside the JSON never closes the block.\n"
        "- After receiving a [Tool result for ...], use that data to answer the user\n\n"
        f"Available tools:\n{tool_spec}"
    )


def _google_tool_choice_instruction(req: dict) -> str:
    """Extract tool_choice constraint from Google API toolConfig."""
    tool_config = req.get("toolConfig", {})
    fc_config = tool_config.get("functionCallingConfig", {})
    mode = fc_config.get("mode", "AUTO")
    allowed = fc_config.get("allowedFunctionNames", [])

    if mode == "NONE":
        return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
    if mode == "ANY":
        if allowed:
            names = ", ".join(f'"{n}"' for n in allowed)
            return f"\n\nIMPORTANT: You MUST call one of these tools: {names}. Do not respond with text only."
        return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
    return ""


def google_contents_to_prompt(req: dict) -> tuple:
    """Convert Google API contents/tools/systemInstruction to (prompt_str, images_list).

    Returns (prompt, images) where images is a list of (bytes, mime_type) tuples.
    """
    parts = []
    images = []

    tool_config = req.get("toolConfig", {})
    fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")

    tools = req.get("tools")
    tool_defs = []
    if tools and fc_mode != "NONE":
        for tool_group in tools:
            for fn in tool_group.get("functionDeclarations", []):
                td = {"name": fn.get("name", ""), "description": fn.get("description", "")}
                params = fn.get("parameters") or fn.get("parametersJsonSchema")
                if params:
                    td["parameters"] = params
                tool_defs.append(td)

    sys_inst = req.get("systemInstruction")
    if sys_inst:
        sys_parts = sys_inst.get("parts", [])
        sys_text = " ".join(p.get("text", "") for p in sys_parts if p.get("text"))
        if sys_text:
            if tool_defs:
                constraint = _google_tool_choice_instruction(req)
                parts.append(sys_text + "\n\n" + build_tool_prompt(tool_defs) + constraint)
            else:
                parts.append(sys_text)
    elif tool_defs:
        constraint = _google_tool_choice_instruction(req)
        parts.append(build_tool_prompt(tool_defs) + constraint)

    for content in req.get("contents", []):
        role = content.get("role", "user")
        msg_parts = []
        for p in content.get("parts", []):
            if p.get("text"):
                msg_parts.append(p["text"])
            elif p.get("inlineData"):
                data = p["inlineData"]
                try:
                    images.append((
                        base64.b64decode(data["data"], validate=True),
                        data.get("mimeType", "image/png"),
                    ))
                    msg_parts.append("[Image attached]")
                except (KeyError, ValueError, TypeError, binascii.Error):
                    pass
            elif p.get("functionCall"):
                fc = p["functionCall"]
                msg_parts.append(
                    f'```function_call\n{json.dumps({"name": fc["name"], "args": fc.get("args", {})}, ensure_ascii=False)}\n```'
                )
            elif p.get("functionResponse"):
                fr = p["functionResponse"]
                msg_parts.append(
                    f'[Tool result for {fr.get("name", "")}]: {json.dumps(fr.get("response", {}), ensure_ascii=False)}'
                )
        text = "\n".join(msg_parts)
        if role == "model":
            parts.append(f"[Assistant]: {text}")
        else:
            parts.append(text)

    return "\n\n".join(p for p in parts if p), images


# A ``function_call`` marker with no fence around it. Models drop the
# backticks often enough that ignoring these would lose real calls. The
# ``(?:^|\\n)`` is deliberate: with ``search(text, pos)`` a bare ``^`` only ever
# matches the very start of the string, so an already-scanned marker is never
# found again and the scan always moves forward.
_BARE_FUNCTION_CALL = re.compile(r'(?:^|\n)function_call\s*\n')


def _iter_bare_function_calls(text: str):
    """Yield ``(start, end, data)`` for each unfenced ``function_call`` marker.

    Same ``raw_decode`` technique as :func:`_iter_tool_blocks`: the payload
    starts right after the marker and ends when the JSON object does, so a
    nested ``args`` object neither truncates it nor confuses it. The regex this
    replaces stopped at the *first* ``}`` -- i.e. one level too early for any
    payload with nested arguments -- and, because the same pattern was used to
    blank the match out of the answer, left the trailing ``}`` behind while the
    call itself was lost.
    """
    dec = json.JSONDecoder(strict=False)
    pos = 0
    while True:
        m = _BARE_FUNCTION_CALL.search(text, pos)
        if not m:
            return
        j = m.end()
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        try:
            data, end = dec.raw_decode(text, j)
        except ValueError:
            # Unreadable payload: skip the marker only, leaving the text as
            # the model wrote it rather than eating a chunk of the answer.
            pos = m.end()
            continue
        if end <= pos:
            pos = m.end()
            continue
        yield m.start(), end, data
        pos = end


def parse_google_function_calls(text: str, allowed_names=None,
                                tool_schemas=None) -> tuple:
    """Extract function_call blocks from model output.

    Handles 3 formats:
    1. ```function_call\\n{...}\\n``` (standard)
    2. function_call\\n{...} (without backticks)
    3. Raw JSON with "name" + "args" keys

    Both the fenced and the unfenced payload are located with
    ``JSONDecoder.raw_decode`` rather than a regex. A non-greedy pattern stops
    at the first fence or brace that appears *inside* the payload -- ordinary
    for an ``args`` object with nested fields, or for a ``write`` call whose
    content is Markdown -- which cuts the JSON in half: ``json.loads`` then
    fails and the pattern used to blank the block removes that much text from
    the answer too. Blocks that still cannot be read are left in the text, so
    nothing the model wrote is silently dropped.

    ``allowed_names``/``tool_schemas`` mirror :func:`parse_tool_calls`:
    names are normalised (``functions.read``/case variants) and undeclared
    or placeholder-copied calls are skipped rather than executed.

    Returns (clean_text, [{"name": ..., "args": ...}])
    """
    from .gemini import log

    function_calls = []

    def _accept(data) -> dict:
        if not (isinstance(data, dict) and data.get("name")):
            return None
        name = _normalize_tool_name(data.get("name"), allowed_names)
        if allowed_names is not None and name is None:
            log(f"Dropping function_call block: undeclared tool "
                f"'{data.get('name')}'")
            return None
        resolved = name or data.get("name")
        props = (tool_schemas or {}).get(resolved) if tool_schemas else None
        args = _arguments_dict(data, props)
        if isinstance(args, dict) and any(
                isinstance(v, str) and v.startswith("EXAMPLE_")
                for v in args.values()):
            log(f"Dropping function_call block: placeholder args copied "
                f"for '{resolved}'")
            return None
        return {"name": resolved, "args": args}
    function_calls = []

    # 1. Fenced blocks. ``json`` fences are skipped: without a declared-name
    #    check (this endpoint parses before it knows what the client sent) an
    #    ordinary JSON example in prose would be taken for a call.
    parts, last_end = [], 0
    for start, end, data, kind, _body in _iter_tool_blocks(text):
        if kind == "json" or not (isinstance(data, dict) and data.get("name")):
            continue
        if kind in ("function_call", "tool_call", "tool_code"):
            line_start = text.rfind("\n", 0, start) + 1
            if text[line_start:start].strip():
                continue
        accepted = _accept(data)
        if accepted is None:
            continue
        function_calls.append(accepted)
        parts.append(text[last_end:start])
        last_end = end
    # The tail stays scannable for stage 2 below (a bare marker may follow a
    # fenced block); it is dropped from the final text when calls exist.
    head, tail = "".join(parts), text[last_end:]
    clean = head + tail

    # 2. The same payload with a bare marker and no fences.
    before_bare = len(function_calls)
    parts, last_end = [], 0
    for start, end, data in _iter_bare_function_calls(clean):
        accepted = _accept(data)
        if accepted is None:
            continue
        function_calls.append(accepted)
        parts.append(clean[last_end:start])
        last_end = end
    if len(function_calls) > before_bare:
        clean = "".join(parts)
    elif before_bare:
        # Only fenced calls: a turn either calls or answers, so nothing after
        # the last block -- prose, or a self-written "[Tool result ...]" --
        # survives.
        clean = head
    else:
        parts.append(clean[last_end:])
        clean = "".join(parts)
    clean = _strip_fabricated(clean, bool(function_calls)).strip()

    # 3. The whole answer is one bare JSON payload.
    if not function_calls and clean.startswith("{"):
        try:
            data = json.loads(clean, strict=False)
        except (json.JSONDecodeError, ValueError):
            data = None
        accepted = _accept(data) if isinstance(data, dict) else None
        if accepted is not None:
            function_calls.append(accepted)
            clean = ""
    return clean, function_calls
