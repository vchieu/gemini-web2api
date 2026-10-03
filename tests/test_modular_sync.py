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
    messages_to_prompt,
    parse_tool_calls,
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

    def test_refuses_non_loopback_without_keys(self):
        CONFIG["api_keys"] = []

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _guard_bind("0.0.0.0", allow_insecure=False)

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


if __name__ == "__main__":
    unittest.main()
