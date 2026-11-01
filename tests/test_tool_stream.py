import threading, unittest
from unittest import mock
from gemini_web2api.config import CONFIG
from gemini_web2api.server import GeminiHandler

# The canonical fence the parser accepts is ```tool_call (see
# tools._FENCE_OPEN: only tool_call/function_call/tool_code/json). A ```bash
# fence parses to NO tool call at all, which previously made these tests hit
# the "model did not emit the required tool call" raise after exhausting
# every retry attempt.
CALL = '```tool_call\n{"name":"read","arguments":{"path":"a"}}\n```'


class ToolStreamTests(unittest.TestCase):
    def setUp(self):
        self.orig = dict(CONFIG)
        CONFIG["tool_retry_on_miss"] = True
        CONFIG["tool_stream"] = True
        # Deterministic: no sleeps between attempts even if a reply misses.
        CONFIG["retry_delay_sec"] = 0
    def tearDown(self):
        CONFIG.clear(); CONFIG.update(self.orig)

    def _params(self, **over):
        params = dict(prompt="p", model_id=2, think_mode=1, file_refs=None,
                      extra=None, tools_active=True, allowed_names={"read"},
                      tool_schemas={"read": {"path"}}, required_tool=False,
                      messages=[{"role": "user", "content": "read file a"}],
                      ticket=None, tool_required={"read": {"path"}},
                      client_gone=threading.Event())
        params.update(over)
        return params

    def _call(self, gen_attr, deltas, **over):
        params = self._params(**over)
        with mock.patch("gemini_web2api.server." + gen_attr,
                        return_value=iter(deltas)) as m:
            return (GeminiHandler.__new__(GeminiHandler)
                    ._generate_with_tool_retry(**params)), m

    def test_openai_loop_uses_generate_stream_when_tool_stream(self):
        (text, calls), m = self._call("generate_stream", [CALL])
        m.assert_called_once()
        self.assertTrue(calls, "tool call should parse from streamed text")

    def test_stream_attempt_exits_early_on_complete_fence(self):
        # generate_stream yields delta-by-delta; once the closing fence has
        # arrived the loop must stop pulling (no waiting for the tail).
        seen = []

        def fake_stream(*a, **k):
            for d in ['```tool_cal', 'l\n{"name":"read","argum',
                      'ents":{"path":"a"}}\n```', "TAIL SHOULD NOT BE READ"]:
                seen.append(d)
                yield d

        with mock.patch("gemini_web2api.server.generate_stream",
                        side_effect=fake_stream):
            result = (GeminiHandler.__new__(GeminiHandler)
                      ._generate_with_tool_retry(**self._params()))
        self.assertTrue(result[1], "call parsed from streamed deltas")
        self.assertNotIn("TAIL SHOULD NOT BE READ", seen,
                         "early-exit must stop reading after the fence closes")

    def test_buffered_path_still_uses_generate(self):
        CONFIG["tool_stream"] = False
        params = self._params()
        with mock.patch("gemini_web2api.server.generate",
                        return_value=CALL) as g, \
             mock.patch("gemini_web2api.server.generate_stream") as s:
            text, calls = (GeminiHandler.__new__(GeminiHandler)
                           ._generate_with_tool_retry(**params))
        g.assert_called_once()
        s.assert_not_called()
        self.assertTrue(calls, "buffered path still parses the tool call")

    def test_ping_interval_reduced(self):
        from gemini_web2api import server
        self.assertLessEqual(server.SSE_PING_INTERVAL_SEC, 5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
