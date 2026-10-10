import http.client
import base64
import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import parse_qs

from gemini_web2api.__main__ import _guard_bind, _maybe_refresh_bl
from gemini_web2api.config import CONFIG, DEFAULT_CONFIG
from gemini_web2api.gemini import (
    _build_headers,
    _build_payload,
    check_routing,
    extract_response_text,
    generate_stream,
    upstream_echo,
)
from gemini_web2api.models import TICKET_HEADER, resolve_model, ticket_for
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import (
    PROMPT_MAX_BYTES,
    google_contents_to_prompt,
    is_required_tool_choice,
    looks_like_missed_tool_call,
    looks_like_tool_call,
    looks_like_upstream_error,
    messages_to_prompt,
    missing_required_params,
    parse_google_function_calls,
    parse_tool_calls,
    pending_tool_request,
    tool_required_params,
)


def _wrb_line(text: str) -> str:
    """Build one upstream ``wrb.fr`` line carrying ``text``.

    Mirrors the shape production parses (``arr[0][2]`` is the inner JSON, whose
    ``[4]`` holds the text parts) and pads past the length guards.
    """
    inner = json.dumps([0, 0, 0, 0, [[None, [text]]]], ensure_ascii=False)
    line = json.dumps([["wrb.fr", None, inner]], ensure_ascii=False)
    return line + " " * max(0, 250 - len(line))


def _wrb_line_parts(texts) -> str:
    """Like ``_wrb_line`` but with several independent text parts in one line."""
    inner = json.dumps([0, 0, 0, 0, [[None, [t]] for t in texts]], ensure_ascii=False)
    line = json.dumps([["wrb.fr", None, inner]], ensure_ascii=False)
    return line + " " * max(0, 250 - len(line))


class _FakeHttpResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        pass

    def iter_text(self):
        return iter(self._chunks)


class _FakeStreamContext:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return _FakeHttpResponse(self._chunks)

    def __exit__(self, *exc):
        return False


class _FakeHttpClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, *args, **kwargs):
        return _FakeStreamContext(self._chunks)


def _decode_payload(payload):
    outer = json.loads(parse_qs(payload)["f.req"][0])
    return json.loads(outer[1])


def _decode_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = next(
            (line[len("event: "):] for line in lines if line.startswith("event: ")),
            None,
        )
        data = next(
            (line[len("data: "):] for line in lines if line.startswith("data: ")),
            None,
        )
        if event_type and data:
            events.append((event_type, json.loads(data)))
    return events


class PayloadPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_temporary_chats_default_to_disabled(self):
        self.assertIs(DEFAULT_CONFIG["temporary_chats"], False)

    def test_persistent_chat_payload(self):
        CONFIG["temporary_chats"] = False

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [2])
        self.assertIsNone(inner[45])

    def test_temporary_chat_payload(self):
        CONFIG["temporary_chats"] = True

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[45], 1)

    def test_payload_includes_uploaded_image_refs(self):
        inner = _decode_payload(_build_payload("describe", 1, 4, ["/uploaded/image-ref"]))

        self.assertEqual(inner[0][0], "describe")
        self.assertEqual(inner[0][3], [[None, None, "/uploaded/image-ref"]])

    def test_request_dump_only_when_debug_raw_enabled(self):
        with mock.patch("gemini_web2api.gemini._dump_raw") as dump:
            CONFIG["debug_raw"] = False
            _build_payload("secret-prompt", 1, 4)
            dump.assert_not_called()

            CONFIG["debug_raw"] = True
            _build_payload("secret-prompt", 1, 4)
            dump.assert_called_once()
            dumped = dump.call_args[0][0]
            self.assertIn("REQUEST >>>", dumped)
            self.assertIn("secret-prompt", dumped)


class ModelRoutingTests(unittest.TestCase):
    """Model selection = inner[79] (family) + inner[80] (variant).

    Both decoded from live browser StreamGenerate captures (Sep 2026). The
    regression: sending inner[79] alone (no variant, no ticket) made the
    upstream ignore the request and answer with the account default, so every
    model -- flash, lite, thinking -- came back as 3.1 Pro.
    """
    def test_browser_captured_family_variant_pairs(self):
        cases = {
            "gemini-3.7-flash": (1, 1),
            "gemini-3.6-flash": (1, 1),
            "gemini-3.5-flash": (1, 1),
            "gemini-3.5-flash-thinking": (2, 2),
            "gemini-3.1-pro": (3, 1),
            "gemini-3.1-pro-enhanced": (3, 3),
            "gemini-auto": (4, 1),
            "gemini-3.5-flash-thinking-lite": (5, 2),
            "gemini-flash-lite": (6, 1),
        }
        for name, (family, variant) in cases.items():
            with self.subTest(model=name):
                _, mode, _, err, extra = resolve_model(name)
                self.assertIsNone(err)
                inner = _decode_payload(_build_payload("hi", mode, 4, extra_fields=extra))
                self.assertEqual(inner[79], family)
                self.assertEqual(inner[80], variant)

    def test_explicit_variant_is_not_overridden(self):
        # pro-enhanced pins inner[80]=3 through extra; the per-model variant
        # must not clobber it.
        _, _, _, err, extra = resolve_model("gemini-3.1-pro-enhanced")
        self.assertIsNone(err)
        self.assertEqual(extra[80], 3)


class ModelTicketTests(unittest.TestCase):
    # The upstream routes BY the X-Goog-Ext-525001261-Jspb ticket and ignores
    # f.req [79]/[80] without it (verified live: (3,1)+pro-ticket -> Pro,
    # (3,1)+flash-ticket -> Flash -- the ticket wins over the body).
    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["model_tickets"] = {
            "flash": '[1,null,null,null,"fbb127bbb056c959",null,null,0,'
                     '[4,5,6,8,4,5,6,8],null,null,1,null,null,1,1,"561701FD",'
                     'null,null,[[6,908199999],[1789884088,624000000]]]',
            "lite-thinking": '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,'
                             '[4,5,6,8,4,5,6,8],null,null,1,null,null,6,2,"279B5F21",'
                             'null,null,[[2,950300000],[1789899759,320000000]]]',
        }

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_ticket_mapping(self):
        self.assertEqual(
            ticket_for("gemini-3.6-flash"), CONFIG["model_tickets"]["flash"])
        self.assertEqual(
            ticket_for("gemini-3.5-flash-thinking-lite"),
            CONFIG["model_tickets"]["lite-thinking"])
        # No ticket configured (or no ticket at all for the model).
        self.assertIsNone(ticket_for("gemini-3.1-pro"))
        self.assertIsNone(ticket_for("gemini-auto"))

    def test_ticket_embeds_family_variant(self):
        flash = json.loads(CONFIG["model_tickets"]["flash"])
        lite_thinking = json.loads(CONFIG["model_tickets"]["lite-thinking"])
        self.assertEqual((flash[14], flash[15]), (1, 1))
        self.assertEqual((lite_thinking[14], lite_thinking[15]), (6, 2))

    def test_ticket_header_sent(self):
        headers = _build_headers(ticket="TICKET-VALUE")
        self.assertEqual(headers[TICKET_HEADER], "TICKET-VALUE")
        headers = _build_headers()
        self.assertNotIn(TICKET_HEADER, headers)

    def test_upstream_echo_parsing(self):
        raw = self._raw_echo("3.5 Flash-Lite", 6, 1)
        self.assertEqual(upstream_echo(raw), ("3.5 Flash-Lite", 6, 1))
        self.assertIsNone(upstream_echo("garbage"))

    def _raw_echo(self, label, family, variant):
        meta = [None] * 60
        meta[42] = label
        meta[58] = family
        meta[59] = variant
        return json.dumps([["wrb.fr", None, json.dumps(meta)]]) + "\n" + "x" * 200

    def test_routing_mismatch_is_logged(self):
        # The observed failure mode: thinking-lite asked for, Pro answered.
        raw = self._raw_echo("3.1 Pro", 3, 1)
        with mock.patch("gemini_web2api.gemini.log") as logger:
            check_routing(raw, model_id=5, extra_fields={80: 2})
        self.assertTrue(any("Routing mismatch" in c.args[0]
                            for c in logger.call_args_list))

    def test_matching_route_is_not_logged(self):
        # The lite-thinking ticket embeds (6,2), so Flash-Lite Extended is the
        # expected answer here even though the body asked for family 5 -- the
        # ticket wins over the body fields.
        raw = self._raw_echo("3.5 Flash-Lite Extended", 6, 2)
        with mock.patch("gemini_web2api.gemini.log") as logger:
            check_routing(raw, model_id=5, extra_fields={80: 2},
                          ticket=CONFIG["model_tickets"]["lite-thinking"])
        self.assertFalse(any("Routing mismatch" in c.args[0]
                             for c in logger.call_args_list))


class MessageParsingTests(unittest.TestCase):
    def test_messages_to_prompt_extracts_openai_image_url_data_url(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_messages_to_prompt_extracts_responses_input_image_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [("https://example.com/image.png", "image/png")])

    def test_messages_to_prompt_ignores_malformed_image_data_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
            ],
        }])

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])

    def test_google_contents_to_prompt_extracts_inline_image_data(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": image_data}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe\n[Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_google_contents_to_prompt_ignores_malformed_inline_image_data(self):
        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": "%%%"}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])


class PromptTruncationTests(unittest.TestCase):
    """Oversized prompts must drop the *middle*, never the pending question."""

    def _shrink_budget(self):
        """Pin the budget at the old 60000 default so these tests exercise
        truncation even though config now ships a much larger prompt_max_bytes.
        """
        return mock.patch.dict(CONFIG, {"prompt_max_bytes": 60000})

    def test_over_long_prompt_keeps_latest_user_message(self):
        filler = "Old history line " * 4000  # ~68 KB, past the 60 KB budget
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": filler},
            {"role": "assistant", "content": filler},
            {"role": "user", "content": "THE FINAL QUESTION"},
        ]

        with self._shrink_budget():
            prompt, _ = messages_to_prompt(messages)

        self.assertIn("THE FINAL QUESTION", prompt)
        self.assertIn("[System instruction]: You are helpful.", prompt)
        self.assertIn("[...truncated...]", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")), 60000)

    def test_over_long_prompt_keeps_tool_definitions(self):
        tools = [
            {"type": "function", "function": {
                "name": f"fn_{i}", "description": "do a thing",
                "parameters": {"type": "object", "properties": {}}}}
            for i in range(400)
        ]
        filler = "Old history line " * 4000
        messages = [
            {"role": "user", "content": filler},
            {"role": "user", "content": "THE FINAL QUESTION"},
        ]

        with self._shrink_budget():
            prompt, _ = messages_to_prompt(messages, tools)

        self.assertIn("# Tool Use", prompt)
        self.assertIn("THE FINAL QUESTION", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")), 60000)

    def test_giant_tool_result_keeps_label_and_older_messages(self):
        # Regression: one 46 KB file read used to be allowed to claim the whole
        # budget. Only its tail reached the model -- no "[Tool result ...]"
        # label, no original request -- so it re-read files it had already read
        # and finally answered the wrong question entirely.
        filler = "def line_of_code(): pass  # " * 2600  # ~73 KB, over the 60 KB budget
        messages = [
            {"role": "user", "content": "REVIEW THE SOURCE AND LIST CRITICAL BUGS"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "read", "arguments": '{"path": "server.py"}'}}]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "[Tool result for read (id=call_1)]: server.py contents\n" + filler},
        ]

        with self._shrink_budget():
            prompt, _ = messages_to_prompt(messages)

        self.assertIn("REVIEW THE SOURCE AND LIST CRITICAL BUGS", prompt)
        self.assertIn("[Tool result for read (id=call_1)]", prompt)
        self.assertIn("[...truncated...]", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")), 60000)

    def test_config_override_raises_budget(self):
        # The fix: default budget comes from config, not a 60 KB constant, so
        # a normal agent conversation (several file reads) is never truncated.
        filler = "Old history line " * 4000  # ~68 KB: over 60000, under 262144
        messages = [
            {"role": "user", "content": filler},
            {"role": "user", "content": "THE FINAL QUESTION"},
        ]

        prompt, _ = messages_to_prompt(messages)

        self.assertNotIn("[...truncated...]", prompt)
        self.assertIn("THE FINAL QUESTION", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")),
                             CONFIG["prompt_max_bytes"])

    def test_prompt_max_bytes_helper_ignores_garbage(self):
        from gemini_web2api.tools import _prompt_max_bytes
        with mock.patch.dict(CONFIG, {"prompt_max_bytes": "not-a-number"}):
            self.assertEqual(_prompt_max_bytes(), PROMPT_MAX_BYTES)
        with mock.patch.dict(CONFIG, {"prompt_max_bytes": 0}):
            self.assertEqual(_prompt_max_bytes(), PROMPT_MAX_BYTES)
        with mock.patch.dict(CONFIG, {"prompt_max_bytes": 1234}):
            self.assertEqual(_prompt_max_bytes(), 1234)

    def test_short_prompt_is_not_truncated(self):
        prompt, _ = messages_to_prompt([{"role": "user", "content": "hello"}])

        self.assertEqual(prompt, "hello")

    def test_tool_block_instructs_literal_triple_backticks(self):
        # An agent-context write arrived with shortened fences
        # (```python -> `python), silently corrupting the file being
        # written. The root cause turned out to be in the test harness
        # rather than the model, but the invariant is cheap to state and
        # expensive to violate, so the block asserts it explicitly.
        tools = [{"type": "function", "function": {
            "name": "write", "description": "create a file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}}}}}]

        prompt, _ = messages_to_prompt([{"role": "user", "content": "hi"}], tools)

        self.assertIn("Triple backticks inside a string argument", prompt)


class RawTraceTests(unittest.TestCase):
    """debug_trace captures every leg of the exchange; off by default."""

    def test_trace_writes_all_legs_when_enabled(self):
        from gemini_web2api.gemini import trace
        fd, path = tempfile.mkstemp(prefix="gw2a-trace-", suffix=".log")
        os.close(fd)
        os.unlink(path)
        try:
            with mock.patch.dict(CONFIG, {"debug_trace": True, "trace_file": path}):
                trace("CLIENT -> PROXY (/v1/chat/completions)", '{"messages": []}')
                trace("PROXY -> MODEL", b"bytes body")
                trace("MODEL -> PROXY", {"inner": 4})
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("CLIENT -> PROXY (/v1/chat/completions)", text)
            self.assertIn('{"messages": []}', text)
            self.assertIn("PROXY -> MODEL", text)
            self.assertIn("bytes body", text)
            self.assertIn("MODEL -> PROXY", text)
            self.assertIn('"inner": 4', text)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_trace_is_silent_when_disabled(self):
        from gemini_web2api.gemini import trace
        fd, path = tempfile.mkstemp(prefix="gw2a-trace-", suffix=".log")
        os.close(fd)
        os.unlink(path)
        try:
            with mock.patch.dict(CONFIG, {"debug_trace": False, "trace_file": path}):
                trace("PROXY -> CLIENT", "should not appear")
            self.assertFalse(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)


class StreamingEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def post_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    def post_chunked_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_stream_starts_with_assistant_role(self, generate_stream):
        generate_stream.return_value = iter(["hel", "lo"])

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"],
                         {"role": "assistant", "content": ""})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hel"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "lo"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value="hi")
    def test_chat_message_and_choice_carry_required_nullable_fields(self, _generate):
        """`logprobs` on the choice is required-nullable, `refusal` is documented
        as null in the spec's own example.

        The Python SDK tolerates absent keys; strict SDKs (Go, Rust, Java)
        generated from the schema do not.
        """
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        self.assertEqual(status, 200)
        choice = json.loads(body)["choices"][0]
        self.assertIn("logprobs", choice)
        self.assertIsNone(choice["logprobs"])
        self.assertIn("refusal", choice["message"])
        self.assertIsNone(choice["message"]["refusal"])

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_stream_chunks_carry_required_nullable_fields(self, generate_stream):
        generate_stream.return_value = iter(["hel", "lo"])

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        self.assertEqual(status, 200)
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        for chunk in chunks:
            for choice in chunk["choices"]:
                self.assertIn("logprobs", choice)
                self.assertIsNone(choice["logprobs"])
        # include_usage: every chunk carries `usage`, null except the last,
        # whose choices list is empty and holds the totals.
        for chunk in chunks[:-1]:
            self.assertIn("usage", chunk)
            self.assertIsNone(chunk["usage"])
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(
            sorted(chunks[-1]["usage"]),
            ["completion_tokens", "prompt_tokens", "total_tokens"],
        )

    @mock.patch("gemini_web2api.server.generate", return_value="chunked ok")
    def test_chat_accepts_chunked_body(self, _generate):
        status, _, body = self.post_chunked_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "chunked ok")

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="looks good")
    def test_chat_accepts_openai_image_url_data_url(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "looks good")

    @mock.patch("gemini_web2api.server.fetch_image_bytes", return_value=b"\xff\xd8\xffremote jpeg")
    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/remote-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="remote ok")
    def test_responses_accepts_input_image_url(self, generate, upload_image, fetch_image_bytes):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is shown?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        fetch_image_bytes.assert_called_once_with("https://example.com/image.jpg")
        upload_image.assert_called_once_with(b"\xff\xd8\xffremote jpeg", "image.png", "image/jpeg")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/remote-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="top-level image ok")
    def test_responses_accepts_top_level_input_image(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "input_text", "text": "What is shown?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                ],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("What is shown?", generate.call_args.args[0])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", side_effect=RuntimeError("upload denied"))
    def test_google_image_upload_failure_returns_502(self, _upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_data,
                        },
                    }],
                }],
            },
        )

        self.assertEqual(status, 502)
        self.assertIn("image upload failed: upload denied", json.loads(body)["error"]["message"])

    @mock.patch("gemini_web2api.server.generate_stream", return_value=iter(["streamed"]))
    def test_google_stream_generate_content_uses_sse(self, _generate_stream):
        status, headers, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:streamGenerateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "Stream this"}],
                }],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn('"text": "streamed"', body)

    @mock.patch("gemini_web2api.server.generate", return_value="hello")
    def test_responses_text_stream_has_complete_event_sequence(self, _generate):
        status, headers, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "hello",
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[4][1]["delta"], "hello")
        self.assertEqual(events[-1][1]["response"]["status"], "completed")
        self.assertEqual(events[-1][1]["response"]["output"][0]["content"][0]["text"], "hello")

    @mock.patch("gemini_web2api.server.generate", return_value="Hello!")
    def test_responses_object_includes_required_fields(self, _generate):
        """The spec marks these required on every response object (nullable,
        but present), and requires the two usage breakdowns by name."""
        status, _, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "hi",
                "instructions": "be terse",
                "temperature": 0.2,
            },
        )

        self.assertEqual(status, 200)
        data = json.loads(body)
        for key in ("id", "object", "created_at", "status", "model", "output",
                    "error", "incomplete_details", "instructions", "tools",
                    "parallel_tool_calls", "metadata", "tool_choice",
                    "temperature", "top_p", "access_programs", "usage"):
            self.assertIn(key, data)
        self.assertIsNone(data["error"])
        self.assertIsNone(data["incomplete_details"])
        self.assertEqual(data["instructions"], "be terse")
        self.assertEqual(data["temperature"], 0.2)
        self.assertEqual(data["tool_choice"], "auto")
        part = data["output"][0]["content"][0]
        self.assertEqual(part["annotations"], [])
        self.assertEqual(part["logprobs"], [])
        self.assertEqual(data["output"][0]["status"], "completed")
        usage = data["usage"]
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(usage["input_tokens_details"]["cache_write_tokens"], 0)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], 0)

    @mock.patch("gemini_web2api.server.generate", return_value="hello")
    def test_responses_text_events_carry_required_logprobs(self, _generate):
        """Both text events require `logprobs`, as does the content part."""
        status, _, body = self.post_json(
            "/v1/responses",
            {"model": "gemini-3.6-flash", "input": "hi", "stream": True},
        )

        self.assertEqual(status, 200)
        events = _decode_sse(body)
        by_type = {event_type: event for event_type, event in events}
        self.assertEqual(by_type["response.output_text.delta"]["logprobs"], [])
        self.assertEqual(by_type["response.output_text.done"]["logprobs"], [])
        self.assertEqual(by_type["response.content_part.added"]["part"]["logprobs"], [])
        self.assertEqual(
            by_type["response.completed"]["response"]["output"][0]["content"][0]["logprobs"],
            [],
        )
        # The in_progress objects carry the same required fields, with a null
        # usage until the response completes.
        created = by_type["response.created"]["response"]
        for key in ("error", "incomplete_details", "instructions", "tools",
                    "parallel_tool_calls", "metadata", "tool_choice",
                    "temperature", "top_p", "access_programs", "usage"):
            self.assertIn(key, created)
        self.assertIsNone(created["usage"])

    @mock.patch("gemini_web2api.server.parse_tool_calls")
    @mock.patch("gemini_web2api.server.generate", return_value="tool output")
    def test_responses_function_call_stream_has_complete_event_sequence(
        self, _generate, parse_tool_calls
    ):
        parse_tool_calls.return_value = (
            "",
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                }
            ],
        )

        status, _, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "weather",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[2][1]["output_index"], 0)
        self.assertEqual(events[3][1]["delta"], '{"city":"Shanghai"}')
        self.assertEqual(events[4][1]["arguments"], '{"city":"Shanghai"}')
        self.assertEqual(events[-1][1]["response"]["output"][0]["name"], "get_weather")


    @mock.patch("gemini_web2api.server.parse_tool_calls")
    @mock.patch("gemini_web2api.server.generate", return_value="get_weather")
    def test_chat_stream_with_tools_emits_indexed_tool_calls(self, _generate, parse_tool_calls):
        parse_tool_calls.return_value = (
            "",
            [{
                "id": "call_test",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
            }],
        )

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        self.assertTrue(tool_chunks)
        self.assertEqual(tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather",
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate_stream", side_effect=RuntimeError("boom"))
    def test_chat_stream_error_before_start_emits_error_event_and_done(self, _generate_stream):
        # The head goes out before upstream starts (a thinking model may take
        # minutes for its first byte), so a failure can no longer be an HTTP
        # status: it arrives as an SSE error event on the already-open stream.
        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        error_frames = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {") and "error" in json.loads(line[len("data: "):])
        ]
        self.assertEqual(len(error_frames), 1)
        self.assertIn("boom", error_frames[0]["error"]["message"])
        self.assertEqual(error_frames[0]["error"]["type"], "api_error")
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate")
    def test_chat_stream_with_tools_headers_before_upstream_finishes(self, generate):
        """The response head must arrive while upstream is still generating.

        OpenCode (and any streaming client) waits for response headers; before
        this, a tools turn buffered everything and the client saw nothing for
        minutes, hit its header timeout, and aborted -- while upstream answers
        kept arriving at a dead socket.
        """
        upstream_started = threading.Event()
        release_upstream = threading.Event()

        def slow_generate(*args, **kwargs):
            upstream_started.set()
            if not release_upstream.wait(timeout=5):
                raise RuntimeError("upstream was never released")
            return "hi"

        generate.side_effect = slow_generate

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }],
                "stream": True,
            }),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = connection.getresponse()
            # The head arrived while generate() is still blocked: the upstream
            # call is in flight and nobody has released it yet.
            self.assertTrue(upstream_started.wait(timeout=2))
            self.assertFalse(release_upstream.is_set())
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
            release_upstream.set()
            body = response.read().decode()
        finally:
            release_upstream.set()
            connection.close()

        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"],
                         {"role": "assistant", "content": ""})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hi"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.SSE_PING_INTERVAL_SEC", 0.05)
    @mock.patch("gemini_web2api.server.generate")
    def test_chat_stream_keepalive_ping_during_slow_upstream(self, generate):
        """An SSE comment must reach the client while upstream still works.

        The comment carries no event (SSE parsers ignore it), but it proves
        the connection is alive so no client idle/chunk timeout fires.
        """
        release_upstream = threading.Event()

        def slow_generate(*args, **kwargs):
            release_upstream.wait(timeout=5)
            return "done"

        generate.side_effect = slow_generate

        body = json.dumps({
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }],
            "stream": True,
        }).encode()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            sock.settimeout(3)
            received = b""
            while b": ping" not in received:
                received += sock.recv(4096)
        finally:
            release_upstream.set()
            sock.close()
        self.assertIn(b": ping\n\n", received)

    def test_chat_stream_midstream_error_emits_error_event_and_done(self):
        def flaky_stream(*args, **kwargs):
            yield "partial"
            raise RuntimeError("boom")

        with mock.patch("gemini_web2api.server.generate_stream", side_effect=flaky_stream):
            status, headers, body = self.post_json(
                "/v1/chat/completions",
                {
                    "model": "gemini-3.6-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn('"error"', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value="")
    def test_chat_empty_upstream_returns_503(self, _generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["type"], "api_error")

    def test_models_endpoint_ignores_query_string(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", "/v1/models?x=1")
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(body)["object"], "list")

    def test_options_preflight_allows_authorization_header(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("OPTIONS", "/v1/chat/completions")
        response = connection.getresponse()
        headers = dict(response.getheaders())
        response.read()
        connection.close()

        self.assertEqual(response.status, 204)
        self.assertIn("Authorization", headers["Access-Control-Allow-Headers"])

    @mock.patch("gemini_web2api.server.generate", return_value="22 Celsius")
    def test_tool_message_links_call_id_to_function_name(self, generate):
        status, _, _ = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [
                    {"role": "user", "content": "weather in Shanghai?"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_abc", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": '{"city":"Shanghai"}'}},
                    ]},
                    {"role": "tool", "tool_call_id": "call_abc", "content": "22C"},
                ],
            },
        )

        self.assertEqual(status, 200)
        prompt = generate.call_args.args[0]
        self.assertIn("[Tool result for get_weather", prompt)
        self.assertIn("id=call_abc", prompt)
        self.assertIn("22C", prompt)

    @mock.patch("gemini_web2api.server.generate", return_value="done")
    def test_responses_function_call_and_output_are_included(self, generate):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "weather?"}]},
                    {"type": "function_call", "call_id": "call_1",
                     "name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                    {"type": "function_call_output", "call_id": "call_1", "output": "22C"},
                ],
            },
        )

        self.assertEqual(status, 200)
        prompt = generate.call_args.args[0]
        self.assertIn("get_weather", prompt)
        self.assertIn("22C", prompt)

    @mock.patch("gemini_web2api.server.generate", return_value="hi")
    def test_null_model_falls_back_without_error(self, _generate):
        status, _, _ = self.post_json(
            "/v1/chat/completions",
            {"model": None, "messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(status, 200)

    @mock.patch("gemini_web2api.server.generate", return_value="hi")
    def test_unknown_model_echoes_requested_name(self, _generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model"], "gpt-4o")

    @mock.patch("gemini_web2api.server.generate", return_value='```json\n{"a": 1}\n```')
    def test_response_format_json_strips_code_fence(self, generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "give json"}],
                "response_format": {"type": "json_object"},
            },
        )

        self.assertEqual(status, 200)
        self.assertIn("JSON", generate.call_args.args[0])
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], '{"a": 1}')

    @mock.patch("gemini_web2api.server.generate", return_value="hi")
    def test_n_greater_than_one_returns_400(self, _generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")

    def test_404_error_uses_openai_shape(self):
        status, _, body = self.post_json("/v1/nonexistent", {"x": 1})

        self.assertEqual(status, 404)
        error = json.loads(body)["error"]
        self.assertIsInstance(error, dict)
        for key in ("message", "type", "param", "code"):
            self.assertIn(key, error)

    @mock.patch("gemini_web2api.server.generate", return_value='```json\n{"a": 1}\n```')
    def test_chat_stream_with_response_format_emits_clean_json(self, _generate):
        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "give json"}],
                "response_format": {"type": "json_object"},
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        content = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks
        )
        self.assertEqual(content, '{"a": 1}')
        self.assertNotIn("```", content)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value='{"a": 1}')
    def test_responses_text_format_json_object_builds_instruction(self, generate):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "give json",
                "text": {"format": {"type": "json_object"}},
            },
        )

        self.assertEqual(status, 200)
        self.assertIn("valid JSON object only", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.generate", return_value="done")
    def test_responses_tool_choice_function_form_builds_instruction(self, generate):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "weather?",
                "tool_choice": {"type": "function", "name": "get_weather"},
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "parameters": {}}}],
            },
        )

        self.assertEqual(status, 200)
        self.assertIn('MUST call the tool "get_weather"', generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.generate", return_value="done")
    def test_responses_function_call_without_arguments_stays_valid_json(self, generate):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "22C"},
                ],
            },
        )

        self.assertEqual(status, 200)
        prompt = generate.call_args.args[0]
        self.assertIn('"arguments": {}', prompt)
        self.assertNotIn('"arguments": }', prompt)


    @mock.patch("gemini_web2api.server.generate", return_value="aaaaSTOPBBBB")
    def test_responses_stop_string_is_applied(self, _generate):
        status, _, body = self.post_json(
            "/v1/responses",
            {"model": "gemini-3.6-flash", "input": "hi", "stop": "STOP"},
        )

        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["output"][0]["content"][0]["text"], "aaaa")
        self.assertEqual(data["status"], "completed")

    @mock.patch("gemini_web2api.server.generate", return_value="x" * 100)
    def test_responses_max_output_tokens_marks_incomplete(self, _generate):
        status, _, body = self.post_json(
            "/v1/responses",
            {"model": "gemini-3.6-flash", "input": "hi", "max_output_tokens": 4},
        )

        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "incomplete")
        self.assertEqual(data["incomplete_details"]["reason"], "max_output_tokens")
        self.assertEqual(len(data["output"][0]["content"][0]["text"]), 16)
        # The message itself was cut off, so it cannot claim to be completed.
        self.assertEqual(data["output"][0]["status"], "incomplete")

    def test_google_stream_midstream_error_emits_error_chunk(self):
        def flaky_stream(*args, **kwargs):
            yield "partial"
            raise RuntimeError("boom")

        with mock.patch("gemini_web2api.server.generate_stream", side_effect=flaky_stream):
            status, headers, body = self.post_json(
                "/v1beta/models/gemini-3.6-flash:streamGenerateContent",
                {"contents": [{"role": "user", "parts": [{"text": "Stream this"}]}]},
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        texts, errors = [], []
        for line in body.splitlines():
            if not line.startswith("data: {"):
                continue
            payload = json.loads(line[len("data: "):])
            if "error" in payload:
                errors.append(payload["error"])
            for candidate in payload.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    texts.append(part.get("text", ""))
        self.assertIn("partial", texts)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["status"], "UNAVAILABLE")

    def test_unexpected_get_error_returns_500(self):
        with mock.patch.object(GeminiHandler, "_needs_auth",
                               side_effect=RuntimeError("boom")):
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            body = response.read().decode()
            connection.close()

        self.assertEqual(response.status, 500)
        self.assertEqual(json.loads(body)["error"]["type"], "api_error")

    # ── Giai đoạn 4: retry when the model talks instead of acting ──

    _READ_TOOL = [{
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    }]

    @mock.patch("gemini_web2api.server.generate")
    def test_chat_retries_when_model_announces_a_tool_it_never_calls(self, generate):
        """A reply that says "let me read..." and stops is not an answer.

        The client sees prose with no tool_calls and treats the turn as final,
        so the session dies after one step. One retry with a hard instruction
        is enough to recover it.
        """
        generate.side_effect = [
            "Dung tool read de minh doc file README:",
            '```tool_call\n{"name": "read", "arguments": {"path": "README.md"}}\n```',
        ]

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "doc README"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 2)
        # Conditional, not "block ONLY": this is a heuristic retry, so a
        # finished answer must still be allowed to stand as prose.
        self.assertIn("Do not describe a tool call",
                      generate.call_args_list[1].args[0])
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(choice["finish_reason"], "tool_calls")

    @mock.patch("gemini_web2api.server.generate",
                return_value="Da tao file xong, ban can gi nua khong?")
    def test_chat_does_not_retry_an_ordinary_short_answer(self, generate):
        """A genuine short reply must not cost a second upstream call."""
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "tao file"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(json.loads(body)["choices"][0]["finish_reason"], "stop")

    @mock.patch("gemini_web2api.server.generate",
                return_value="Dung tool read de minh doc file:")
    def test_chat_retry_can_be_switched_off(self, generate):
        CONFIG["tool_retry_on_miss"] = False

        status, _, _ = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "doc README"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 1)

    @mock.patch("gemini_web2api.server.generate")
    def test_chat_retries_when_the_model_answers_about_an_unread_file(self, generate):
        """A fluent reply about a file nothing ever opened is a fabrication.

        No fence, no refusal, no awkward phrasing -- the response-side
        heuristic misses it completely (the text is longer than its length
        guard too). The only thing that is provably false is that no tool call
        in the conversation ever touched ``README.md``, and Opencode would
        otherwise stop here and end the session on the invention.
        """
        fabricated = (
            "Duoi day la 3 diem chinh thuong co trong tai lieu `README.md` cua "
            "cac du an dang Gemini Web2API. " * 10)
        generate.side_effect = [
            fabricated,
            '```tool_call\n{"name": "read", "arguments": {"path": "README.md"}}\n```',
        ]

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user",
                           "content": "Doc README.md va ke 3 diem chinh."}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        # Long enough that the response-side length guard rules it out: this
        # turn is retried purely because the file was never read.
        self.assertGreater(len(fabricated), 600)
        self.assertEqual(generate.call_count, 2)
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "read")

    @mock.patch("gemini_web2api.server.generate")
    def test_chat_does_not_retry_once_the_file_was_read(self, generate):
        """Summarising something already read is a real answer, not a miss."""
        generate.return_value = "Tong ket: README.md noi du an nay lam gi."

        status, _, _ = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [
                 {"role": "user", "content": "Doc README.md roi tong ket."},
                 {"role": "assistant", "content": "", "tool_calls": [{
                     "id": "c1", "type": "function",
                     "function": {"name": "read",
                                  "arguments": '{"path": "README.md"}'}}]},
                 {"role": "tool", "content": "# README ...",
                  "tool_call_id": "c1"},
                 {"role": "user", "content": "Doc README.md roi tong ket nua."},
             ],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 1)

    @mock.patch("gemini_web2api.server.generate")
    def test_chat_retries_an_empty_upstream_response(self, generate):
        """A body that never arrived is not an answer -- nor "all done" either.

        A connection dropped mid-generation leaves ``""`` behind; returning
        that as 200 tells Opencode the turn finished with nothing to do, so it
        stops instead of letting the model try again.
        """
        generate.side_effect = [
            "",
            '```tool_call\n{"name": "read", "arguments": {"path": "README.md"}}\n```',
        ]

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "doc README"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 2)
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "read")

    @mock.patch("gemini_web2api.server.generate")
    def test_empty_response_without_tools_is_reported_not_retried(self, generate):
        """With no tools there is no block to chase, so a blank reply is a 503."""
        generate.return_value = ""

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "xin chao"}]},
        )

        self.assertEqual(status, 503)
        self.assertEqual(generate.call_count, 1)
        self.assertIn("empty response", body)

    @mock.patch("gemini_web2api.server.generate",
                return_value="I encountered an error doing what you asked. "
                             "Could you try again?")
    def test_upstream_error_placeholder_is_retried_then_reported(self, generate):
        """Gemini's canned failure sentence must never reach the client as 200.

        It reads as a finished answer, so the operator sees "the model replied
        with an error" instead of a failure they can act on.
        """
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "doc README"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 503)
        self.assertEqual(generate.call_count, 2)
        error = json.loads(body)["error"]
        self.assertEqual(error["type"], "api_error")
        self.assertIn("placeholder", error["message"])

    @mock.patch("gemini_web2api.server.generate")
    def test_chat_retries_when_the_tool_call_omits_a_required_parameter(self, generate):
        """A call that parses but cannot run is not a call the client can use.

        ``{"name": "read", "arguments": {}}`` is valid on the wire, so the
        client executes it, its tool fails on the missing path and the error
        comes back as a whole extra round-trip -- at the end of which the
        model, shown only its own complete-looking call, still does not know
        which field was absent. The retry names it.
        """
        generate.side_effect = [
            '```tool_call\n{"name": "read", "arguments": {}}\n```',
            '```tool_call\n{"name": "read", "arguments": {"path": "README.md"}}\n```',
        ]

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "doc README"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("missing required parameter(s): read.path",
                      generate.call_args_list[1].args[0])
        choice = json.loads(body)["choices"][0]
        self.assertEqual(
            json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
            {"path": "README.md"},
        )

    @mock.patch("gemini_web2api.server.generate")
    def test_extra_tool_retries_are_configurable(self, generate):
        """One retry leaves a forgetful model failing on a real share of turns.

        The count of extra attempts belongs to the config so an operator can
        trade upstream calls for reliability instead of it being fixed here.
        """
        CONFIG["tool_retry_attempts"] = 2
        generate.return_value = "Dung tool read de minh doc file:"

        status, _, _ = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "doc README"}],
             "tools": self._READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 3)

    @mock.patch("gemini_web2api.server.generate")
    def test_google_generate_retries_when_the_model_announces_a_call(self, generate):
        """The native endpoint had no retry, so Gemini CLI got the prose.

        Everything the OpenAI endpoints do to catch "I will read that..." with
        no block behind it applies here too: this is the same upstream and the
        same model.
        """
        generate.side_effect = [
            "Dung tool read de minh doc file:",
            '```function_call\n{"name": "read", "args": {"path": "README.md"}}\n```',
        ]

        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "doc README"}]}],
             "tools": [{"functionDeclarations": [{
                 "name": "read",
                 "parameters": {"type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"]}}]}]},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 2)
        parts = json.loads(body)["candidates"][0]["content"]["parts"]
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        self.assertEqual(calls, [{"name": "read", "args": {"path": "README.md"}}])

    @mock.patch("gemini_web2api.server.generate",
                return_value="I encountered an error doing what you asked. "
                             "Could you try again?")
    def test_google_generate_reports_the_upstream_placeholder_as_an_error(self, generate):
        """The canned failure sentence used to reach the client as a candidate.

        It reads as a finished answer, so the operator sees "the model replied
        with an error" instead of a failure they can act on.
        """
        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
             "tools": [{"functionDeclarations": [{"name": "read"}]}]},
        )

        self.assertEqual(status, 503)
        self.assertEqual(generate.call_count, 2)
        error = json.loads(body)["error"]
        self.assertEqual(error["status"], "UNAVAILABLE")
        self.assertIn("placeholder", error["message"])

    # ── Giai đoạn 6: /v1/responses has to behave like /v1/chat/completions ──

    _RESPONSES_READ_TOOL = [{
        "type": "function",
        "name": "read",
        "description": "Read a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    }]

    @mock.patch("gemini_web2api.server.generate",
                return_value='```tool_call\n{"arguments": {"path": "README.md"}}\n```')
    def test_responses_infers_a_missing_tool_name(self, generate):
        """The Responses path passed no schemas, so an unnamed call was dropped.

        The model omits ``name`` often enough that losing the schemas cost real
        calls on this endpoint while the Chat Completions path recovered them.
        """
        status, _, body = self.post_json(
            "/v1/responses",
            {"model": "gemini-3.6-flash", "input": "doc README",
             "tools": self._RESPONSES_READ_TOOL},
        )

        self.assertEqual(status, 200)
        calls = [o for o in json.loads(body)["output"] if o["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "README.md"})

    @mock.patch("gemini_web2api.server.generate")
    def test_responses_retries_when_model_announces_a_tool_it_never_calls(self, generate):
        generate.side_effect = [
            "Dung tool read de minh doc file:",
            '```tool_call\n{"name": "read", "arguments": {"path": "README.md"}}\n```',
        ]

        status, _, body = self.post_json(
            "/v1/responses",
            {"model": "gemini-3.6-flash", "input": "doc README",
             "tools": self._RESPONSES_READ_TOOL},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 2)
        calls = [o for o in json.loads(body)["output"] if o["type"] == "function_call"]
        self.assertEqual(calls[0]["name"], "read")

    @mock.patch("gemini_web2api.server.generate", return_value="plain text, no call")
    def test_responses_flattened_tool_choice_still_requires_a_call(self, generate):
        """The Responses API names the target without nesting it under ``function``.

        Reading only ``tool_choice["function"]`` made this shape look like plain
        ``auto``, so the required-tool retry never ran for Responses clients.
        """
        status, _, _ = self.post_json(
            "/v1/responses",
            {"model": "gemini-3.6-flash", "input": "doc README",
             "tools": self._RESPONSES_READ_TOOL,
             "tool_choice": {"type": "function", "name": "read"}},
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_count, 2)

    # ── Unsupported methods must still answer in JSON ──

    def test_delete_unknown_response_returns_json_404(self):
        """DELETE /v1/responses/{id} is a spec endpoint; nothing is stored here.

        The status matches the spec's 404 for an unknown id, and the body is
        an OpenAI error object instead of BaseHTTPRequestHandler's bare-text
        501.
        """
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("DELETE", "/v1/responses/resp_missing")
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()

        self.assertEqual(response.status, 404)
        error = json.loads(body)["error"]
        for key in ("message", "type", "param", "code"):
            self.assertIn(key, error)
        self.assertEqual(error["code"], "response_not_found")

    def test_get_unknown_response_returns_json_404(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", "/v1/responses/resp_missing")
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()

        self.assertEqual(response.status, 404)
        self.assertEqual(json.loads(body)["error"]["code"], "response_not_found")

        # Deeper paths are not a response id, so they get the generic 404.
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", "/v1/responses/resp_missing/input_items")
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()

        self.assertEqual(response.status, 404)
        self.assertNotEqual(json.loads(body)["error"]["code"], "response_not_found")

    def test_put_and_patch_return_json_405(self):
        for method in ("PUT", "PATCH"):
            with self.subTest(method=method):
                connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                connection.request(method, "/v1/chat/completions", body=b"{}",
                                   headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                body = response.read().decode()
                connection.close()

                self.assertEqual(response.status, 405)
                self.assertEqual(response.getheader("Content-Type"), "application/json")
                error = json.loads(body)["error"]
                self.assertEqual(error["code"], "method_not_allowed")
                self.assertEqual(error["type"], "invalid_request_error")


class GenerateStreamTests(unittest.TestCase):
    """Upstream line framing for generate_stream (scaffolding vs real code)."""

    def _run(self, chunks):
        with mock.patch("gemini_web2api.gemini.HAS_HTTPX", True), \
                mock.patch("gemini_web2api.gemini._get_httpx_client",
                           return_value=_FakeHttpClient(chunks)), \
                mock.patch("gemini_web2api.gemini._get_url",
                           return_value="http://example.invalid"), \
                mock.patch("gemini_web2api.gemini._build_headers", return_value={}):
            return list(generate_stream("q", 1, 4))

    def test_plain_code_fence_streams_before_it_closes(self):
        fence = chr(96) * 3
        # The fence is still open when this line arrives. Buffering on backtick
        # parity would have withheld the whole block until the closing fence.
        out = self._run([_wrb_line(f"Sure thing:\n{fence}python\nprint('hi')\n") + "\n"])

        self.assertTrue(out)
        self.assertIn(f"{fence}python", out[0])

    def test_upstream_scaffolding_block_is_stripped_whole(self):
        fence = chr(96) * 3
        block = f"{fence}python?code_reference&code_event_index=0\nprint(1)\n{fence}"

        out = self._run([_wrb_line(f"ok\n{block}\ndone") + "\n"])

        self.assertEqual(out, ["ok\ndone"])
        self.assertNotIn("code_reference", out[0])

    def test_unrelated_text_parts_in_one_line_are_all_emitted(self):
        out = self._run([
            _wrb_line_parts(["Reasoning summary. " * 10, "The answer is 42."]) + "\n"
        ])

        self.assertEqual(len(out), 2)
        self.assertEqual(out[1], "The answer is 42.")


class ResponseExtractionTests(unittest.TestCase):
    """Which text of a multi-text payload becomes the response body.

    ``generate_stream`` already treats ``texts[1:]`` in one line as separate
    blocks, so the shape is known to occur. The non-streaming path reaches the
    same place by taking the longest text it can find -- which drops a tool
    call whenever the model also says anything of length.
    """

    CALL = '```tool_call\n{"name": "read", "arguments": {"path": "a.py"}}\n```'

    def test_a_tool_call_wins_over_a_longer_summary(self):
        summary = "Dai hon nhieu: " + "mo ta viec da doc file a.py. " * 12
        self.assertGreater(len(summary), len(self.CALL))

        raw = _wrb_line_parts([summary, self.CALL])

        self.assertEqual(extract_response_text(raw).strip(), self.CALL.strip())

    def test_without_a_tool_call_length_still_decides(self):
        """Nothing changes for the common case: no candidate looks like a call."""
        long = "Day la cau tra loi day du va chi tiet hon nhieu. " * 6
        raw = _wrb_line_parts(["ngan.", long])

        self.assertEqual(extract_response_text(raw).strip(), long.strip())

    def test_json_quoted_in_prose_is_not_mistaken_for_a_call(self):
        """A fenced example carrying ``name`` must not win the tie.

        Preferring candidates that only *look* like a call would swap a real
        answer for the snippet someone quoted while explaining something.
        """
        self.assertFalse(looks_like_tool_call('```json\n{"name": "Alice"}\n```'))

    def test_a_fenced_json_call_is_still_recognised(self):
        """``{"name": ..., "arguments": ...}`` is a call whatever the fence."""
        self.assertTrue(looks_like_tool_call(
            '```json\n{"name": "read", "arguments": {"path": "a.py"}}\n```'))
        self.assertTrue(looks_like_tool_call(self.CALL))
        self.assertFalse(looks_like_tool_call("```tool_call\nnot json\n```"))


class StartupGuardTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_auto_generates_key_for_non_loopback_without_keys(self):
        CONFIG["api_keys"] = []

        with contextlib.redirect_stderr(io.StringIO()):
            _guard_bind("0.0.0.0", allow_insecure=False)

        # Auto-generated key is required for every request
        self.assertTrue(CONFIG.get("api_keys"))
        self.assertEqual(len(CONFIG["api_keys"]), 1)
        self.assertIsInstance(CONFIG["api_keys"][0], str)
        self.assertEqual(len(CONFIG["api_keys"][0]), 64)

    def test_allows_non_loopback_with_keys(self):
        CONFIG["api_keys"] = ["secret"]

        _guard_bind("0.0.0.0", allow_insecure=False)

    def test_allows_non_loopback_with_allow_insecure(self):
        CONFIG["api_keys"] = []

        _guard_bind("0.0.0.0", allow_insecure=True)

    def test_loopback_always_allowed(self):
        CONFIG["api_keys"] = []

        _guard_bind("127.0.0.1", allow_insecure=False)

    def test_auto_update_bl_refreshes_by_default(self):
        CONFIG["auto_update_bl"] = True
        CONFIG["gemini_bl"] = "old_bl"

        with mock.patch("gemini_web2api.__main__.fetch_latest_bl",
                        return_value="new_bl") as fetch:
            _maybe_refresh_bl()

        fetch.assert_called_once()
        self.assertEqual(CONFIG["gemini_bl"], "new_bl")

    def test_auto_update_bl_false_pins_build_label(self):
        CONFIG["auto_update_bl"] = False
        CONFIG["gemini_bl"] = "pinned_bl"

        with mock.patch("gemini_web2api.__main__.fetch_latest_bl",
                        return_value="new_bl") as fetch:
            _maybe_refresh_bl()

        fetch.assert_not_called()
        self.assertEqual(CONFIG["gemini_bl"], "pinned_bl")


class HostCheckTests(unittest.TestCase):
    """`_check_host` blocks DNS rebinding without mangling `host:port` forms."""

    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    @staticmethod
    def check(host, bind="127.0.0.1", keys=None):
        handler = GeminiHandler.__new__(GeminiHandler)
        handler.headers = {} if host is None else {"Host": host}
        CONFIG["host"] = bind
        CONFIG["api_keys"] = list(keys or [])
        return handler._check_host()

    def test_loopback_with_port_is_allowed(self):
        self.assertTrue(self.check("127.0.0.1:8081"))

    def test_ipv6_loopback_with_port_is_allowed(self):
        # rsplit(":", 1) used to leave "[::1]" behind, which never matched "::1".
        self.assertTrue(self.check("[::1]:8081"))

    def test_localhost_with_port_is_allowed(self):
        self.assertTrue(self.check("localhost:8081"))

    def test_foreign_host_is_rejected(self):
        self.assertFalse(self.check("attacker.example"))
        self.assertFalse(self.check("attacker.example:8081"))

    def test_wildcard_bind_serves_every_interface(self):
        # A LAN IP or container hostname must pass when the operator bound
        # 0.0.0.0 on purpose; only the loopback names matched before.
        self.assertTrue(self.check("192.168.1.10:8081", bind="0.0.0.0"))
        self.assertTrue(self.check("gemini-web2api:8081", bind="0.0.0.0"))

    def test_loopback_bind_rejects_the_wildcard_host(self):
        self.assertFalse(self.check("0.0.0.0:8081", bind="127.0.0.1"))

    def test_configured_key_makes_the_host_check_redundant(self):
        self.assertTrue(self.check("attacker.example", keys=["secret"]))

    def test_missing_host_header_is_allowed(self):
        self.assertTrue(self.check(None))


class ToolParsingTests(unittest.TestCase):
    def test_parse_tool_calls_handles_single_line_block(self):
        clean, calls = parse_tool_calls(
            '```tool_call {"name": "foo", "arguments": {"x": 1}}```')

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "foo")
        self.assertEqual(calls[0]["function"]["arguments"], '{"x": 1}')
        self.assertEqual(clean, "")

    def test_parse_tool_calls_rejects_undeclared_function(self):
        clean, calls = parse_tool_calls(
            '```tool_call\n{"name": "bogus", "arguments": {}}\n```', allowed_names={"real"})

        self.assertEqual(calls, [])
        self.assertIn("bogus", clean)

    def test_parse_tool_calls_keeps_unparsable_block(self):
        text = "before\n```tool_call\nnot json\n```\nafter"

        clean, calls = parse_tool_calls(text, allowed_names={"known"})

        self.assertEqual(calls, [])
        self.assertIn("not json", clean)
        self.assertIn("before", clean)
        self.assertIn("after", clean)

    def test_parse_tool_calls_passes_through_string_arguments(self):
        _, calls = parse_tool_calls(
            '```tool_call\n{"name": "foo", "arguments": "{\\"a\\": 1}"}\n```',
            allowed_names={"foo"},
        )

        self.assertEqual(calls[0]["function"]["arguments"], '{"a": 1}')

    def test_parse_tool_calls_accepts_flattened_arguments(self):
        """The observed production shape: parameters beside ``name``, not nested.

        Gemini writes ``{"name": "run_commands", "commands": [...]}`` often
        enough; reading only ``arguments``/``args`` silently turned every such
        call into ``{}``, which clients reject with "Invalid input".
        """
        clean, calls = parse_tool_calls(
            '```tool_call\n{"name": "run_commands", "commands": ["npm test"]}\n```',
            allowed_names={"run_commands"},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["arguments"], '{"commands": ["npm test"]}')
        self.assertEqual(clean, "")

    def test_parse_tool_calls_accepts_argument_aliases(self):
        for alias, payload in (
            ("args", '{"name": "foo", "args": {"x": 1}}'),
            ("input", '{"name": "foo", "input": {"x": 1}}'),
            ("parameters", '{"name": "foo", "parameters": {"x": 1}}'),
        ):
            with self.subTest(alias=alias):
                _, calls = parse_tool_calls(
                    f"```tool_call\n{payload}\n```", allowed_names={"foo"})

                self.assertEqual(calls[0]["function"]["arguments"], '{"x": 1}')

    def test_parse_tool_calls_keeps_name_only_meta_keys_out_of_arguments(self):
        _, calls = parse_tool_calls(
            '```tool_call\n{"name": "foo", "description": "run it", "x": 1}\n```',
            allowed_names={"foo"},
        )

        self.assertEqual(calls[0]["function"]["arguments"], '{"x": 1}')

    def test_parse_tool_calls_recovers_python_literal_arguments(self):
        _, calls = parse_tool_calls(
            "```tool_call\n{\"name\": \"foo\", \"arguments\": \"{'x': 1}\"}\n```",
            allowed_names={"foo"},
        )

        self.assertEqual(calls[0]["function"]["arguments"], '{"x": 1}')

    def test_parse_tool_calls_degrades_unusable_string_to_empty_object(self):
        """`function.arguments` must always parse as JSON.

        Passing a non-JSON string through made every client fail before it
        could even report which parameter was missing.
        """
        _, calls = parse_tool_calls(
            '```tool_call\n{"name": "foo", "arguments": "npm test"}\n```',
            allowed_names={"foo"},
        )

        self.assertEqual(calls[0]["function"]["arguments"], "{}")

    def test_parse_tool_calls_drops_call_without_name(self):
        clean, calls = parse_tool_calls(
            '```tool_call\n{"commands": ["npm test"]}\n```', allowed_names={"foo"})

        self.assertEqual(calls, [])
        self.assertIn("commands", clean)

    # ── Giai đoạn 1: fences the old non-greedy regex could not survive ──

    def test_parse_tool_calls_survives_fence_inside_arguments(self):
        """A ``write`` call whose content is Markdown/code must not be cut.

        The payload contains ``` inside a JSON string; a regex that stops at
        the first fence halved the JSON, ``json.loads`` failed, and the whole
        call was silently dropped -- leaving the client with prose and no tool.
        """
        payload = json.dumps({
            "name": "write",
            "arguments": {"path": "notes.md",
                          "content": "# Title\n```python\nprint(1)\n```\nafter"},
        })
        clean, calls = parse_tool_calls(f"```tool_call\n{payload}\n```",
                                        allowed_names={"write"})

        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertIn("```python", args["content"])
        self.assertTrue(args["content"].endswith("after"))
        self.assertEqual(clean, "")

    def test_parse_tool_calls_accepts_function_call_fence(self):
        clean, calls = parse_tool_calls(
            '```function_call\n{"name": "read", "args": {"path": "a.py"}}\n```',
            allowed_names={"read"},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")
        self.assertEqual(calls[0]["function"]["arguments"], '{"path": "a.py"}')
        self.assertEqual(clean, "")

    def test_parse_tool_calls_accepts_json_fence_for_declared_tool(self):
        clean, calls = parse_tool_calls(
            '```json\n{"name": "read", "arguments": {"path": "a.py"}}\n```',
            allowed_names={"read"},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")
        self.assertEqual(clean, "")

    def test_parse_tool_calls_keeps_json_fence_for_undeclared_tool(self):
        """A JSON block naming a tool the client never declared is prose."""
        clean, calls = parse_tool_calls(
            '```json\n{"name": "bogus", "arguments": {}}\n```',
            allowed_names={"read"},
        )

        self.assertEqual(calls, [])
        self.assertIn("bogus", clean)

    def test_parse_tool_calls_leaves_plain_json_example_alone(self):
        """JSON quoted while explaining something must never become a call."""
        text = 'Here is the config:\n```json\n{"port": 8081}\n```\nDone.'

        clean, calls = parse_tool_calls(text, allowed_names={"read", "write"})

        self.assertEqual(calls, [])
        self.assertIn('{"port": 8081}', clean)

    def test_parse_tool_calls_ignores_unfenced_json(self):
        """Bare JSON with no fence is not a call -- the format needs a fence."""
        text = '{"name": "read", "arguments": {"path": "a.py"}}'

        clean, calls = parse_tool_calls(text, allowed_names={"read"})

        self.assertEqual(calls, [])
        self.assertIn('"read"', clean)

    def test_a_literal_newline_inside_a_string_argument_survives(self):
        """``write`` payloads are file bodies, and files are multi-line.

        Strict JSON rejects a raw control character inside a string, so the
        whole block used to be reported as "not valid JSON" and the call
        silently turned into prose. The value handed on must come back escaped.
        """
        text = (
            '```tool_call\n'
            '{"name": "write", "arguments": '
            '{"path": "a.md", "content": "# Test\n- one\n- two"}}'
            '\n```'
        )

        clean, calls = parse_tool_calls(text, allowed_names={"write"})

        self.assertEqual(len(calls), 1)
        self.assertNotIn("tool_call", clean)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["content"], "# Test\n- one\n- two")

    def test_a_literal_newline_inside_a_string_arguments_value_survives(self):
        """The same, one level down: ``arguments`` handed over as a string."""
        text = (
            '```tool_call\n'
            '{"name": "write", "arguments": '
            '"{\\"path\\": \\"a.md\\", \\"content\\": \\"x\ny\\"}"}'
            '\n```'
        )

        _, calls = parse_tool_calls(text, allowed_names={"write"})

        self.assertEqual(len(calls), 1)
        # Re-serialised, so the client gets valid JSON rather than the raw
        # newline it would choke on.
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["content"], "x\ny")


class MissedToolCallDetectionTests(unittest.TestCase):
    """Giai đoạn 4 heuristics: when is prose a stalled first turn?"""

    def test_announcement_before_a_colon_is_a_miss(self):
        self.assertTrue(looks_like_missed_tool_call("Dung tool read de doc file:"))

    def test_refusal_and_let_me_read_phrases_are_misses(self):
        for text in (
            "I cannot access the filesystem from here.",
            "Let me read the README first.",
            "De minh doc file README truoc.",
            "Để mình đọc file README trước.",
            "Ban khong co quyen truy cap thu muc nay.",
            "Bạn không có quyền truy cập thư mục này.",
        ):
            with self.subTest(text=text):
                self.assertTrue(looks_like_missed_tool_call(text))

    def test_long_answer_is_never_a_miss(self):
        text = "Let me read " + "noi dung rat dai. " * 60
        self.assertGreater(len(text), 600)
        self.assertFalse(looks_like_missed_tool_call(text))

    def test_response_with_a_tool_fence_is_not_a_miss(self):
        self.assertFalse(looks_like_missed_tool_call(
            'Da doc xong.\n```tool_call\n{"name":"read"}\n```'))

    def test_ordinary_short_completion_is_not_a_miss(self):
        self.assertFalse(looks_like_missed_tool_call("Da tao file xong."))
        self.assertFalse(looks_like_missed_tool_call(""))

    def test_upstream_placeholder_is_recognised(self):
        self.assertTrue(looks_like_upstream_error(
            "I encountered an error doing what you asked. Could you try again?"))
        self.assertFalse(looks_like_upstream_error(
            "I encountered an error while writing the docs about how errors work."))
        self.assertFalse(looks_like_upstream_error(""))

    def test_required_tool_choice_recognises_both_api_shapes(self):
        self.assertTrue(is_required_tool_choice("required"))
        self.assertTrue(is_required_tool_choice(
            {"type": "function", "function": {"name": "read"}}))
        self.assertTrue(is_required_tool_choice(
            {"type": "function", "name": "read"}))
        self.assertFalse(is_required_tool_choice("auto"))
        self.assertFalse(is_required_tool_choice("none"))
        self.assertFalse(is_required_tool_choice({"type": "auto"}))
        self.assertFalse(is_required_tool_choice(None))


class PendingToolRequestTests(unittest.TestCase):
    """A fabricated answer about an unread file is provably wrong.

    The reply itself reads perfectly -- no fence, no refusal -- so the only
    signal is that no tool call in the conversation ever opened the file being
    discussed. Opencode then stops, ending the session on an invention.
    """

    @staticmethod
    def _read_readme():
        return [
            {"role": "user", "content": "Doc README.md roi tong ket."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read", "arguments": '{"path": "README.md"}'}}]},
            {"role": "tool", "content": "# README ...", "tool_call_id": "c1"},
            {"role": "user", "content": "Doc README.md nua di."},
        ]

    def test_unread_file_requested_by_the_user_is_reported(self):
        reason = pending_tool_request(
            [{"role": "user", "content": "Doc README.md va ke 3 diem chinh."}])
        self.assertIn("README.md", reason)

    def test_a_file_already_read_is_not_reported(self):
        self.assertIsNone(pending_tool_request(self._read_readme()))

    def test_sub_path_still_counts_as_a_read_of_the_basename(self):
        messages = [
            {"role": "user", "content": "Doc gemini_web2api/server.py"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function", "function": {
                    "name": "read",
                    "arguments": '{"path": "gemini_web2api/server.py"}'}}]},
            {"role": "tool", "content": "...", "tool_call_id": "c1"},
            {"role": "user", "content": "Doc gemini_web2api/server.py nua."},
        ]
        self.assertIsNone(pending_tool_request(messages))

    def test_tool_demand_with_no_call_at_all_is_reported(self):
        reason = pending_tool_request(
            [{"role": "user", "content": "Hay dung tool de xem ket qua."}])
        self.assertIn("demanded", reason)

    def test_tool_demand_is_ignored_once_a_call_happened(self):
        self.assertIsNone(pending_tool_request([
            {"role": "user", "content": "Hay dung tool de xem ket qua."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read", "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "content": "...", "tool_call_id": "c1"},
        ]))

    def test_a_call_that_names_no_file_still_counts_as_tool_use(self):
        """``glob {"pattern": "**/*"}`` proves the model is acting.

        Its arguments contain no file-shaped token, so keying "tool use
        happened" off the file names would report a phantom demand and burn
        a retry on a turn that is already doing what the user asked.
        """
        self.assertIsNone(pending_tool_request([
            {"role": "user",
             "content": "Review source code thu muc. Bat buoc dung tool."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function", "function": {
                    "name": "glob",
                    "arguments": '{"path": "src", "pattern": "**/*"}'}}]},
            {"role": "tool", "content": "a.py\nb.py", "tool_call_id": "c1"},
        ]))

    def test_runtime_names_are_not_mistaken_for_files(self):
        """`Node.js` matches the ``.js`` pattern but no tool ever opens it.

        Without the exclusion, every turn that mentions a runtime would be
        reported as owed a read that can never happen, and each of those turns
        would be retried.
        """
        for msg in ("Cai dat Node.js roi chay du an.",
                    "Du an nay dung Next.js va Vue.js.",
                    "Minh da dung Express.js truoc do."):
            with self.subTest(msg=msg):
                self.assertIsNone(pending_tool_request([{"role": "user",
                                                         "content": msg}]))

    def test_the_same_file_spelled_differently_counts_as_read(self):
        """Users type ``readme.md``; ``read`` is handed ``README.md``.

        The two name one file -- on Windows, literally -- so a case-sensitive
        difference must not look like an unread request. The turn ends on a
        user message so the tool-call guard cannot answer this by itself.
        """
        self.assertIsNone(pending_tool_request([
            {"role": "user", "content": "Doc readme.md truoc."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"path": "README.md"}'}}]},
            {"role": "tool", "content": "# README ...", "tool_call_id": "c1"},
            {"role": "user", "content": "Doc readme.md roi tong ket nhe."},
        ]))

    def test_a_tool_call_this_turn_ends_the_check(self):
        """After the model has acted, whatever it writes is a real answer.

        Even if it names a file that call did not spell out -- the summary of
        a review legitimately mentions files it never listed -- chasing it
        would replace a finished answer with the retry's output.
        """
        self.assertIsNone(pending_tool_request([
            {"role": "user",
             "content": "Review README.md va server.py trong du an."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"path": "README.md"}'}}]},
            {"role": "tool", "content": "# README ...", "tool_call_id": "c1"},
            {"role": "user", "content": "Tiep tuc review server.py di."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c2", "type": "function",
                "function": {"name": "glob",
                             "arguments": '{"pattern": "**/*"}'}}]},
            {"role": "tool", "content": "server.py\nmodels.py",
             "tool_call_id": "c2"},
        ]))

    def test_plain_chitchat_and_empty_input_report_nothing(self):
        self.assertIsNone(pending_tool_request(
            [{"role": "user", "content": "Cam on ban nhieu nhe."}]))
        self.assertIsNone(pending_tool_request([]))
        self.assertIsNone(pending_tool_request(None))

    def test_parse_google_function_calls_accepts_flattened_args(self):
        clean, calls = parse_google_function_calls(
            '```function_call\n{"name": "run_commands", "commands": ["dir"]}\n```')

        self.assertEqual(calls, [{"name": "run_commands", "args": {"commands": ["dir"]}}])
        self.assertNotIn("function_call", clean)

    def test_parse_google_function_calls_always_returns_object_args(self):
        _, calls = parse_google_function_calls(
            '```function_call\n{"name": "foo", "arguments": "broken"}\n```')

        self.assertEqual(calls, [{"name": "foo", "args": {}}])

    def test_messages_to_prompt_states_arguments_nesting(self):
        prompt, _ = messages_to_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {
                "name": "run_commands", "description": "run",
                "parameters": {"type": "object", "properties": {
                    "commands": {"type": "array"}}}}}],
        )

        self.assertIn('{"name": "func_name", "arguments": {"param": "value"}}', prompt)
        self.assertIn('never beside "name"', prompt)
        self.assertIn('"commands"', prompt)


class GoogleFunctionCallParsingTests(unittest.TestCase):
    """The Google endpoint used to locate calls with regexes.

    Everything the OpenAI path already did with ``raw_decode`` applied here as
    well: a payload is found by parsing it, not by guessing where it ends.
    """

    def test_bare_marker_with_nested_args_is_extracted(self):
        """The old pattern stopped at the *first* ``}`` -- one level early.

        It then blanked that half-open match out of the answer, so the client
        got no call at all and the text kept nothing but a stray ``}``.
        """
        clean, calls = parse_google_function_calls(
            'function_call\n{"name": "run_commands", '
            '"args": {"commands": ["dir"], "cwd": "."}}')

        self.assertEqual(calls, [{"name": "run_commands",
                                  "args": {"commands": ["dir"], "cwd": "."}}])
        self.assertNotIn("run_commands", clean)
        self.assertNotIn("}", clean)

    def test_a_fence_inside_a_string_argument_does_not_cut_the_call(self):
        """A ``write`` payload carries a file body, newline and fences and all.

        Strict ``json.loads`` rejects the raw newline the model wrote instead
        of escaping it -- and the pattern that blanked the match took a chunk
        of the answer with it.
        """
        payload = ('{"name": "write_file", "args": {"path": "x.md", '
                   '"content": "# Title\n\n```python\nprint(1)\n```"}}')
        text = "Truoc\n```function_call\n" + payload + "\n```\nSau"

        clean, calls = parse_google_function_calls(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"]["content"],
                         "# Title\n\n```python\nprint(1)\n```")
        self.assertNotIn("function_call", clean)
        self.assertIn("Truoc", clean)
        self.assertIn("Sau", clean)

    def test_unreadable_block_stays_in_the_text(self):
        """Nothing the model wrote may be deleted on the way to a parse error."""
        text = "Truoc\n```function_call\nnot json\n```\nSau"

        clean, calls = parse_google_function_calls(text)

        self.assertEqual(calls, [])
        for kept in ("Truoc", "not json", "Sau"):
            self.assertIn(kept, clean)

    def test_a_fence_and_a_bare_marker_in_one_answer(self):
        clean, calls = parse_google_function_calls(
            '```function_call\n{"name": "read", "args": {"path": "a.md"}}\n```\n'
            'func tiep:\nfunction_call\n{"name": "run", "args": {"cmd": "ls"}}')

        self.assertEqual([c["name"] for c in calls], ["read", "run"])
        self.assertNotIn("function_call", clean)


class ToolRequiredParamTests(unittest.TestCase):
    """A call the client cannot run must be caught before it is sent."""

    TOOLS = [{
        "type": "function",
        "function": {"name": "read",
                     "parameters": {"type": "object",
                                    "properties": {"path": {"type": "string"},
                                                   "mode": {"type": "string"}},
                                    "required": ["path"]}},
    }, {
        "type": "function",
        "function": {"name": "ping",
                     "parameters": {"type": "object", "properties": {}}},
    }]

    def test_tool_required_params_reads_the_required_list(self):
        self.assertEqual(tool_required_params(self.TOOLS), {"read": {"path"}})

    def test_missing_required_params_names_the_tool_and_the_field(self):
        calls = [
            {"type": "function", "function": {"name": "read", "arguments": "{}"}},
            {"type": "function",
             "function": {"name": "read", "arguments": '{"path": "a.md"}'}},
            {"type": "function", "function": {"name": "ping", "arguments": "{}"}},
        ]

        self.assertEqual(
            missing_required_params(calls, tool_required_params(self.TOOLS)),
            ["read.path"],
        )

    def test_nothing_is_reported_without_schemas_or_arguments(self):
        self.assertEqual(
            missing_required_params([{"function": {"name": "x"}}], None), [])
        # Not a JSON object: already reported as unusable arguments instead.
        self.assertEqual(
            missing_required_params(
                [{"function": {"name": "read", "arguments": "[1]"}}],
                {"read": {"path"}}),
            [])


class ToolRetryClientGoneTests(unittest.TestCase):
    """A hung-up client must not be billed another upstream call."""

    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def _handler(self):
        return GeminiHandler.__new__(GeminiHandler)

    @mock.patch("gemini_web2api.server.generate")
    def test_disconnect_during_first_attempt_skips_the_retry(self, generate):
        CONFIG["tool_retry_on_miss"] = True
        client_gone = threading.Event()

        def answer_then_client_leaves(*args, **kwargs):
            # The client hung up while this call was in flight (a keep-alive
            # write failed) and the reply is prose with no tool call.
            client_gone.set()
            return "README.md được mô tả trong AGENTS.md"

        generate.side_effect = answer_then_client_leaves

        text, tool_calls = self._handler()._generate_with_tool_retry(
            "review source giùm", 2, 1, None, None,
            True, {"read"}, {"read": []}, False,
            messages=[{"role": "user", "content": "review source giùm"}],
            client_gone=client_gone,
        )

        # The prose answer is still returned (nothing raises), but the
        # heuristic retry -- a second upstream call nobody will read -- is not.
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(text, "README.md được mô tả trong AGENTS.md")
        self.assertFalse(tool_calls)

    @mock.patch("gemini_web2api.server.generate")
    def test_connected_client_still_gets_the_retry(self, generate):
        CONFIG["tool_retry_on_miss"] = True
        # The first reply is prose (no call); the retry -- an unconnected
        # client would have skipped -- yields a real tool_call block.
        generate.side_effect = [
            "Xem thì rõ",
            '```tool_call\n{"name": "read", "arguments": {"path": "a.py"}}\n```',
        ]

        text, tool_calls = self._handler()._generate_with_tool_retry(
            "review source giùm", 2, 1, None, None,
            True, {"read"}, {"read": []}, False,
            messages=[{"role": "user", "content": "review source giùm"}],
            client_gone=threading.Event(),
        )

        self.assertEqual(generate.call_count, 2)
        self.assertTrue(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "read")


class HttpClientThreadSafetyTests(unittest.TestCase):
    """A client built twice by two racing threads is one nobody ever closes."""

    def test_concurrent_first_calls_build_exactly_one_client(self):
        import gemini_web2api.gemini as gemini

        if not gemini.HAS_HTTPX:
            self.skipTest("httpx is not installed")
        sentinel = object()
        built = []
        real = gemini._httpx_client
        gemini._httpx_client = None
        try:
            with mock.patch.object(gemini.httpx, "Client",
                                   side_effect=lambda **kw: built.append(kw) or sentinel):
                barrier = threading.Barrier(8)
                results = []

                def worker():
                    barrier.wait()
                    results.append(gemini._get_httpx_client())

                threads = [threading.Thread(target=worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10)
        finally:
            gemini._httpx_client = real

        self.assertEqual(len(results), 8)
        self.assertEqual(len(built), 1)
        self.assertTrue(all(r is sentinel for r in results))


if __name__ == "__main__":
    unittest.main()
