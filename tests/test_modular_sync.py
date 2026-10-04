import http.client
import base64
import contextlib
import io
import json
import threading
import unittest
from unittest import mock
from urllib.parse import parse_qs

from gemini_web2api.__main__ import _guard_bind, _maybe_refresh_bl
from gemini_web2api.config import CONFIG, DEFAULT_CONFIG
from gemini_web2api.gemini import _build_payload, generate_stream
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import (
    PROMPT_MAX_BYTES,
    google_contents_to_prompt,
    is_required_tool_choice,
    looks_like_missed_tool_call,
    looks_like_upstream_error,
    messages_to_prompt,
    parse_google_function_calls,
    parse_tool_calls,
    pending_tool_request,
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

    def test_over_long_prompt_keeps_latest_user_message(self):
        filler = "Old history line " * 4000  # ~68 KB, past PROMPT_MAX_BYTES
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": filler},
            {"role": "assistant", "content": filler},
            {"role": "user", "content": "THE FINAL QUESTION"},
        ]

        prompt, _ = messages_to_prompt(messages)

        self.assertIn("THE FINAL QUESTION", prompt)
        self.assertIn("[System instruction]: You are helpful.", prompt)
        self.assertIn("[...truncated...]", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")), PROMPT_MAX_BYTES)

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

        prompt, _ = messages_to_prompt(messages, tools)

        self.assertIn("# Tool Use", prompt)
        self.assertIn("THE FINAL QUESTION", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")), PROMPT_MAX_BYTES)

    def test_short_prompt_is_not_truncated(self):
        prompt, _ = messages_to_prompt([{"role": "user", "content": "hello"}])

        self.assertEqual(prompt, "hello")


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
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hel"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "lo"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

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
    def test_chat_stream_error_before_start_returns_json_502(self, _generate_stream):
        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 502)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("error", json.loads(body))

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
    def test_chat_empty_upstream_returns_502(self, _generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 502)
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
        """With no tools there is no block to chase, so a blank reply is a 502."""
        generate.return_value = ""

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "xin chao"}]},
        )

        self.assertEqual(status, 502)
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

        self.assertEqual(status, 502)
        self.assertEqual(generate.call_count, 2)
        error = json.loads(body)["error"]
        self.assertEqual(error["type"], "api_error")
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


if __name__ == "__main__":
    unittest.main()
