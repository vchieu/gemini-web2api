import threading, unittest
from unittest import mock
from gemini_web2api.config import CONFIG
from gemini_web2api.server import GeminiHandler

class ToolStreamTests(unittest.TestCase):
    def setUp(self):
        self.orig = dict(CONFIG)
        CONFIG["tool_retry_on_miss"] = True
        CONFIG["tool_stream"] = True
    def tearDown(self):
        CONFIG.clear(); CONFIG.update(self.orig)

    def _call(self, gen_attr, deltas, messages=None):
        params = dict(prompt="p", model_id=2, think_mode=1, file_refs=None,
                      extra=None, tools_active=True, allowed_names={"read"},
                      tool_schemas={"read": {"path"}}, required_tool=True,
                      messages=messages or [{"role":"user","content":"review src"}],
                      ticket=None, tool_required={"read": {"path"}},
                      client_gone=threading.Event())
        with mock.patch("gemini_web2api.server." + gen_attr, return_value=iter(deltas)) as m:
            return GeminiHandler.__new__(GeminiHandler)._generate_with_tool_retry(**params), m

    def test_openai_loop_uses_generate_stream_when_tool_stream(self):
        (text, calls), m = self._call("generate_stream", ["```bash\n{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}\n```"])
        m.assert_called_once()
        self.assertTrue(calls, "tool call should parse from streamed text")

    def test_buffered_path_still_uses_generate(self):
        CONFIG["tool_stream"] = False
        params = dict(prompt="p", model_id=2, think_mode=1, file_refs=None,
                      extra=None, tools_active=True, allowed_names={"read"},
                      tool_schemas={"read": {"path"}}, required_tool=True,
                      messages=[{"role":"user","content":"review src"}],
                      ticket=None, tool_required={"read": {"path"}},
                      client_gone=threading.Event())
        with mock.patch("gemini_web2api.server.generate", return_value="```bash\n{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}\n```") as g, \
             mock.patch("gemini_web2api.server.generate_stream") as s:
            GeminiHandler.__new__(GeminiHandler)._generate_with_tool_retry(**params)
        g.assert_called_once(); s.assert_not_called()

    def test_ping_interval_reduced(self):
        from gemini_web2api import server
        self.assertLessEqual(server.SSE_PING_INTERVAL_SEC, 5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
