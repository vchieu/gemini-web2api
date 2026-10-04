"""Tool calling and multimodal message parsing."""
import json
import re
import uuid
import base64
import binascii
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


# Sits between the dropped leading block and the kept tail so the model can see
# that history was removed rather than silently wondering what it missed.
_TRUNCATION_MARKER = "[...truncated...]"


def _join_prompt_parts(parts: list, max_bytes: int) -> str:
    """Join message parts into a prompt, dropping the *middle* when too long.

    Slicing the assembled prompt from the front (the obvious way) keeps the
    system instruction and the oldest history while throwing away the newest
    user message -- i.e. exactly the question that needs answering. Instead we
    keep the leading block (tool definitions / system instruction) and as many
    of the newest messages as fit, and mark the gap.
    """
    items = [p for p in parts if p]

    def nbytes(text: str) -> int:
        return len(text.encode("utf-8"))

    full = "\n\n".join(items)
    if nbytes(full) <= max_bytes:
        return full

    from .gemini import log
    log(f"Prompt truncated to {max_bytes} bytes")

    # Two separators around the marker.
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

    if head:
        return "\n\n".join(head + [_TRUNCATION_MARKER] + tail)
    return "\n\n".join([_TRUNCATION_MARKER] + tail)


def messages_to_prompt(messages: list, tools: list = None, tool_choice=None) -> tuple:
    """Convert OpenAI messages to (prompt_str, images_list).

    Returns (prompt, images) where images is a list of (bytes, mime_type) tuples.
    """
    parts = []
    images = []

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
            # If still too large, slim down by removing parameters
            if len(tools_json.encode("utf-8")) > PROMPT_MAX_BYTES // 3:
                slim_defs = [{"name": t["name"], "description": t["description"]} for t in tool_defs]
                tools_json = json.dumps(slim_defs, ensure_ascii=False)
            parts.append(
                "# Tool Use\n\n"
                "You can call the following tools. Call format:\n"
                '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
                "When calling tools, output ONLY the tool_call block(s).\n\n"
                f"Available tools:\n{tools_json}"
                f"{constraint}"
            )

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

    return _join_prompt_parts(parts, PROMPT_MAX_BYTES), images


def parse_tool_calls(text: str, allowed_names=None) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list).

    ``allowed_names`` optionally restricts accepted function names. Blocks that
    fail to parse, or that name an undeclared function, are left in the text so
    that no content is silently lost.
    """
    tool_calls = []
    pattern = r'```tool_call[ \t]*\n?(.*?)\n?```'
    clean_parts = []
    last_end = 0
    for m in re.finditer(pattern, text, re.DOTALL):
        body = m.group(1).strip()
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            data = None
        parsed = None
        if isinstance(data, dict):
            name = data.get("name")
            if name and (allowed_names is None or name in allowed_names):
                args = data.get("arguments", data.get("args", {}))
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                parsed = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
        if parsed is None:
            continue
        clean_parts.append(text[last_end:m.start()])
        last_end = m.end()
        tool_calls.append(parsed)
    clean_parts.append(text[last_end:])
    clean = "".join(clean_parts).strip()
    return clean, tool_calls


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


def parse_google_function_calls(text: str) -> tuple:
    """Extract function_call blocks from model output.

    Handles 3 formats:
    1. ```function_call\\n{...}\\n``` (standard)
    2. function_call\\n{...} (without backticks)
    3. Raw JSON with "name" + "args" keys

    Returns (clean_text, [{"name": ..., "args": ...}])
    """
    function_calls = []
    pattern1 = r'```function_call\s*\n(.*?)\n```'
    pattern2 = r'(?:^|\n)function_call\s*\n(\{[^`]*?\})'
    clean = text
    for pattern in [pattern1, pattern2]:
        for match in re.findall(pattern, clean, re.DOTALL):
            try:
                data = json.loads(match.strip())
                if "name" in data:
                    function_calls.append({
                        "name": data["name"],
                        "args": data.get("args", data.get("arguments", {})),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        clean = re.sub(pattern, '', clean, flags=re.DOTALL).strip()
    if not function_calls and clean.strip().startswith("{"):
        try:
            data = json.loads(clean.strip())
            if "name" in data and ("args" in data or "arguments" in data):
                function_calls.append({
                    "name": data["name"],
                    "args": data.get("args", data.get("arguments", {})),
                })
                clean = ""
        except (json.JSONDecodeError, KeyError):
            pass
    return clean, function_calls
