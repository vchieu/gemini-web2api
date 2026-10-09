"""Tool calling and multimodal message parsing."""
import ast
import json
import re
import uuid
import base64
import binascii
import unicodedata
from urllib.parse import unquote_to_bytes

# Upper bound for the generated prompt. Keeps large tool lists / long histories
# from being rejected by the upstream web endpoint.
PROMPT_MAX_BYTES = 60000


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


def _join_prompt_parts(parts: list, max_bytes: int, pinned_head: str = None) -> str:
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
    """
    items = [p for p in parts if p]

    def nbytes(text: str) -> int:
        return len(text.encode("utf-8"))

    # Cap the pin so a pathological tool list cannot consume the budget on its
    # own; ~40% leaves the majority for the history and the pending question.
    pinned = ""
    if pinned_head:
        cap = max_bytes * 2 // 5
        pinned = (pinned_head if nbytes(pinned_head) <= cap
                  else pinned_head.encode("utf-8")[:cap].decode("utf-8", errors="ignore"))
        pinned = pinned.strip()
        if not pinned:
            pinned = ""

    full = "\n\n".join(([pinned] if pinned else []) + items)
    if nbytes(full) <= max_bytes:
        return full

    from .gemini import log
    log(f"Prompt truncated to {max_bytes} bytes")
    if pinned:
        log(f"Tool block pinned: {nbytes(pinned)} bytes "
            f"({100 * nbytes(pinned) // max_bytes}% of budget)")

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

    # 1. Keep the newest messages first: they carry the pending question and
    #    the latest tool results.
    tail = []
    used = 0
    for part in reversed(items):
        if not tail:
            if nbytes(part) > budget:
                # Keep the tail (end) of oversized messages, not the head (beginning)
                # This preserves questions at the end of user messages
                part = part.encode("utf-8")[-budget:].decode("utf-8", errors="ignore")
                tail.append(part)
                used = budget
                break
            tail.append(part)
            used = nbytes(part)
            continue
        cost = nbytes(part) + 2
        if used + cost > budget:
            break
        tail.append(part)
        used += cost
    tail.reverse()

    # 2. Spend whatever is left on the leading block (tool defs, system, ...).
    #    This only ever adds a prefix, so history in the middle stays dropped.
    remaining = budget - used
    head = []
    head_used = 0
    for part in items[: len(items) - len(tail)]:
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
    """
    p = t.get("parameters") or {}
    props = p.get("properties") if isinstance(p, dict) else None
    props = props if isinstance(props, dict) else {}
    required = set(p.get("required", [])) if isinstance(p, dict) else set()
    args = ", ".join(
        f'{k}{"" if k in required else "?"}: '
        f'{(v or {}).get("type", "any") if isinstance(v, dict) else "any"}'
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


def _tool_use_block(tools_json: str, constraint: str) -> str:
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
        '{"name": "func_name", "arguments": {"param": "value"}}\n'
        "```\n\n"
        "When calling tools:\n"
        "- Output ONLY tool_call block(s): no prose, explanation or apology "
        "before or after them.\n"
        '- Every parameter belongs inside the "arguments" object, keyed exactly '
        'like the "parameters" properties below -- never beside "name".\n'
        "- Emit several blocks in one turn when the calls are independent.\n"
        "- Never state that you have read a file, run a command or inspected "
        "the code unless a [Tool result for ...] message proves it. Knowing a "
        "file's contents or a command's output requires calling the tool.\n"
        "- Never claim the tools are unavailable, restricted, or that you lack "
        "permission: they are connected and will run.\n"
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
            if tool_bytes > PROMPT_MAX_BYTES // 3:
                for desc_max in (300, 150, 80):
                    tools_json = "\n".join(_compact_tool(t, desc_max)
                                           for t in tool_defs)
                    tool_bytes = len(tools_json.encode("utf-8"))
                    if tool_bytes <= PROMPT_MAX_BYTES // 3:
                        break
                log(f"Tool definitions compacted to signatures "
                    f"(descriptions cut to {desc_max} chars, {tool_bytes} bytes): "
                    "parameter names and types kept")
            else:
                log(f"Tool definitions: {len(tool_defs)} tools, {tool_bytes} bytes")
            tool_block = _tool_use_block(tools_json, constraint)

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
                    name = next_msg.get("name") or id_to_name.get(tcid, "")
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
            name = msg.get("name") or id_to_name.get(tcid, "")
            body = _stringify_content(content)
            parts.append(f"[Tool result for {name} (id={tcid})]: {body}")
        else:
            parts.append(_stringify_content(content))

    if tool_block:
        parts.append(_TOOL_REMINDER)
    return _join_prompt_parts(parts, PROMPT_MAX_BYTES, pinned_head=tool_block), images


# Keys that describe the call itself rather than its parameters. Models emit
# flattened payloads (`{"name": "run_commands", "commands": [...]}`) as readily
# as the canonical nested one, so every other key counts as a parameter.
_CALL_META_KEYS = {"name", "id", "type", "description", "tool", "tool_name"}

# Aliases used for the nested parameter object.
_ARGUMENT_KEYS = ("arguments", "args", "input")

_NOTHING = object()


def extract_arguments(data: dict):
    """Pull a tool call's parameters out of a parsed payload.

    Handles the canonical ``{"name": ..., "arguments": {...}}`` shape and the
    variants models actually emit: ``args`` / ``input`` aliases, a
    ``parameters`` wrapper, and the flattened form where the parameters sit
    beside ``name``. Returns ``{}`` only when the payload carries none.
    """
    for key in _ARGUMENT_KEYS:
        value = data.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            return value
    rest = {k: v for k, v in data.items() if k not in _CALL_META_KEYS}
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
            return json.dumps(recovered, ensure_ascii=False), None
        return "{}", "not a JSON object"
    if args is None:
        return "{}", "missing"
    if isinstance(args, (list, int, float, bool)):
        return json.dumps(args, ensure_ascii=False), None
    return "{}", f"unsupported type {type(args).__name__}"


def _arguments_dict(data: dict) -> dict:
    """``extract_arguments`` for callers that need a JSON object, always."""
    args = extract_arguments(data)
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


# Fences that may wrap a tool call. ``json`` is listed too because models reach
# for it routinely, but it is validated far more strictly below: an ordinary
# JSON example in prose must never be mistaken for a real call.
_FENCE_OPEN = re.compile(r'```(tool_call|function_call|tool_code|json)'
                         r'(?![A-Za-z0-9_])[ \t]*\n?')
_FENCE_CLOSE = re.compile(r'\s*```')


def _iter_tool_blocks(text: str):
    """Yield ``(start, end, data, kind)`` for each tool-call-shaped fenced block.

    The payload is located with ``JSONDecoder.raw_decode`` starting right after
    the opening fence, so a ``` that appears *inside* a string argument (the
    normal shape of a ``write`` call whose content is Markdown or code)
    terminates neither the payload nor the block. A non-greedy regex instead
    stops at the very first fence, cuts the JSON in half, and the call is lost.

    ``data`` is ``_NOTHING`` for a block that declares a tool-call fence but
    whose body is not valid JSON; it is yielded anyway so the caller can log the
    drop rather than silently leaving it in the text.

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
            yield m.start(), (close + 3 if close != -1 else m.end()), _NOTHING, kind
            pos = span_end if close != -1 else m.end()
            continue
        c = _FENCE_CLOSE.match(text, end)
        if c:
            end = c.end()
        yield m.start(), end, data, kind
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
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list).

    ``allowed_names`` optionally restricts accepted function names.
    ``tool_schemas`` maps each declared name to its parameter names; when the
    model omits ``name`` a call is recovered only if its parameters point at
    exactly one declared tool. Blocks that fail to parse, or that end up
    unnamed, are left in the text so that no content is silently lost.
    """
    from .gemini import log

    tool_calls = []
    clean_parts = []
    last_end = 0
    for start, end, data, kind in _iter_tool_blocks(text):
        if kind == "json" and not _is_declared_call(data, allowed_names, tool_schemas):
            # Ordinary JSON in prose: leave it untouched and do not log it.
            continue
        reason = ""
        parsed = None
        if isinstance(data, dict):
            name = data.get("name")
            if not name:
                name = _infer_tool_name(data, tool_schemas)
                if name:
                    log(f"Inferred missing tool name '{name}' from its parameters")
            if not name:
                reason = "no tool name"
            elif allowed_names is not None and name not in allowed_names:
                reason = f"undeclared tool '{name}'"
            else:
                args, degraded = _arguments_json(extract_arguments(data))
                if degraded:
                    log(f"tool_call '{name}': arguments unusable ({degraded}), "
                        "sending an empty object so the client can report it")
                parsed = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
        elif data is _NOTHING:
            reason = "body is not valid JSON"
        else:
            reason = "unrecognised payload"
        if parsed is None:
            log(f"Dropping tool_call block: {reason}: {text[start:end][:300]}")
            continue
        clean_parts.append(text[last_end:start])
        last_end = end
        tool_calls.append(parsed)
    clean_parts.append(text[last_end:])
    clean = "".join(clean_parts).strip()
    return clean, tool_calls


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
    for _start, _end, data, kind in _iter_tool_blocks(text):
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

# Phrases that mean "I am about to look at that" or "I cannot reach that" --
# i.e. the model announced an action it never performed because it never
# emitted the tool call.
_MISSED_TOOL_RE = re.compile(
    r"(khong\s+the\s+truy\s+cap|khong\s+co\s+quyen|khong\s+duoc\s+truy\s+cap"
    r"|khong\s+tim\s+thay|khong\s+the\s+doc"
    r"|cannot\s+access|can'?t\s+access"
    r"|don'?t\s+have\s+access|do\s+not\s+have\s+access"
    r"|no\s+(?:file\s+|tool\s+)?access"
    r"|not\s+(?:permitted|allowed)\s+to\s+read"
    r"|de\s+minh\s+doc|de\s+minh\s+kiem\s+tra"
    r"|let\s+me\s+(?:read|check|look|see|examine|inspect)"
    r"|i(?:'ll|\s+will)\s+(?:read|check|look|see|examine|inspect))",
    re.IGNORECASE,
)

# Anything at or above this is a real answer rather than a stalled first turn.
_MAX_MISSED_TOOL_CHARS = 600


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


# File-shaped tokens a sentence can point at: `README.md`, `server.py`,
# `gemini-web2api.log`. Used on both sides of the check below.
_FILE_REF_RE = re.compile(
    r"(?<![\w.-])[\w][\w.-]*\.(?:"
    r"md|py|pyi|js|jsx|ts|tsx|json|jsonc|txt|ya?ml|toml|cfg|conf|ini|env|"
    r"java|kt|c|cc|cpp|h|hpp|go|rs|rb|php|html|css|scss|less|sql|sh|bash|ps1|"
    r"xml|csv|lock|log|rst|bat|cmd|vue|svelte|proto|gradle|properties"
    r")(?![\w-])",
    re.IGNORECASE,
)

# Names that end in a source extension but are runtimes, frameworks or
# libraries rather than files. `Node.js` and friends would otherwise look like
# an unread file for every turn that mentions them -- no tool call ever
# "opens" them, so the request would be retried until the conversation ends.
_NON_FILE_NAMES = frozenset({
    "node.js", "node.mjs", "node.cjs", "next.js", "nuxt.js", "vue.js",
    "react.js", "react-dom.js", "express.js", "jquery.js", "d3.js",
    "three.js", "ember.js", "backbone.js", "socket.js", "deno.js",
})

# "dùng tool", "đọc file", "review source", "use the tool", ... -- whitespace
# is stripped before matching because folding Vietnamese silently glues words
# together ("sử dụng" -> "sudung") while other clients leave the space.
_TOOL_DEMAND_RE = re.compile(
    r"(?:dung|sudung|haydung|phaidung|batbuocdung|vanphaidung|phairadung)"
    r"(?:la)?(?:tool|tools|congcu|caccongcu)"
    r"|(?:doc|kiemtra|xem|mo)(?:rai)?(?:file|tep|thumuc|cacfile)"
    r"|(?:review|kiemtra|doc)(?:la)?(?:source|code|masnguon|cacfile)"
    r"|(?:use|call|run|invoke)(?:the)?s?(?:tool|tools)"
    r"|(?:read|open|view|check|inspect)(?:the)?s?(?:file|files)",
    re.IGNORECASE,
)


def _files_in(*chunks) -> set:
    """File-shaped names mentioned anywhere in ``chunks`` (strings or JSON)."""
    blob = " ".join(c for c in chunks if isinstance(c, str))
    return {m.group(0) for m in _FILE_REF_RE.finditer(blob)}


def _conversation_called_files(messages) -> set:
    """Every file name any tool call in this conversation has touched."""
    called = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            called |= _files_in(str(fn.get("name") or ""), args)
    return called


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

    # A tool call after the newest user message means this turn already acted.
    if any(isinstance(msg, dict) and msg.get("tool_calls")
           for msg in messages[last_user_idx + 1:]):
        return None
    if not last_user:
        return None

    called_lower = {f.lower() for f in _conversation_called_files(messages)}
    uncovered = sorted(
        f for f in _files_in(last_user)
        if f.lower() not in called_lower and f.lower() not in _NON_FILE_NAMES
    )
    if uncovered:
        return "requested file(s) never read: " + ", ".join(uncovered)

    folded = _fold_diacritics(last_user).replace(" ", "").replace("\n", "")
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
        "- Output ONLY the function_call block(s), nothing else\n"
        "- You may call multiple tools with multiple blocks\n"
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


def parse_google_function_calls(text: str) -> tuple:
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

    Returns (clean_text, [{"name": ..., "args": ...}])
    """
    function_calls = []

    # 1. Fenced blocks. ``json`` fences are skipped: without a declared-name
    #    check (this endpoint parses before it knows what the client sent) an
    #    ordinary JSON example in prose would be taken for a call.
    parts, last_end = [], 0
    for start, end, data, kind in _iter_tool_blocks(text):
        if kind == "json" or not (isinstance(data, dict) and data.get("name")):
            continue
        function_calls.append({"name": data["name"], "args": _arguments_dict(data)})
        parts.append(text[last_end:start])
        last_end = end
    parts.append(text[last_end:])
    clean = "".join(parts)

    # 2. The same payload with a bare marker and no fences.
    parts, last_end = [], 0
    for start, end, data in _iter_bare_function_calls(clean):
        if not (isinstance(data, dict) and data.get("name")):
            continue
        function_calls.append({"name": data["name"], "args": _arguments_dict(data)})
        parts.append(clean[last_end:start])
        last_end = end
    parts.append(clean[last_end:])
    clean = "".join(parts).strip()

    # 3. The whole answer is one bare JSON payload.
    if not function_calls and clean.startswith("{"):
        try:
            data = json.loads(clean, strict=False)
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("name"):
            function_calls.append({
                "name": data["name"],
                "args": _arguments_dict(data),
            })
            clean = ""
    return clean, function_calls
