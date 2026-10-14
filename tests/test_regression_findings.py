"""Regression tests for the static-review findings (tool/evidence/prompt layer).

Each class maps to one finding in the review:

1.  repeat guard vs. implement loop (``_repeatable_calls`` / ``_google_repeatable``)
2.  evidence counting (``files_read_in_turn`` / ``_files_in`` / ``listed_turn_files``)
3.  hard-coded ``source_review_instruction`` removed from the proxy
4.  missed-tool detector covers implement verbs; single 1200 threshold
5.  first user message (plan) pinned across truncation
6.  creation intent vs. read nudge; narrowed source-review heuristic
7.  Google path parity (names / required params / allowlist)
8.  file-ref false positives + path support
9.  mid-sentence fences are not calls
10. fenced markers survive ``_strip_fabricated``
11. squeezed tool pin keeps the constraint
12. ``tool_choice: required`` raises when nothing was emitted
13. ``EXAMPLE_`` placeholder copies are dropped
14. ``extract_arguments`` keeps a declared string ``input``/``args`` param
16. tool-call logging redacts file bodies
"""
import http.client
import json
import threading
import unittest
from unittest import mock

from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import GeminiError
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import (
    _MAX_MISSED_TOOL_CHARS,
    _files_in,
    _google_repeatable,
    _is_source_review,
    _join_prompt_parts,
    _repeatable_calls,
    _strip_fabricated,
    change_demanded,
    extract_arguments,
    files_read_in_turn,
    listed_turn_files,
    looks_like_missed_tool_call,
    messages_to_prompt,
    parse_google_function_calls,
    parse_tool_calls,
    pending_tool_request,
)


def _tc(tcid, name, args):
    return {"id": tcid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _tool_msg(tcid, content, name=""):
    msg = {"role": "tool", "tool_call_id": tcid, "content": content}
    if name:
        msg["name"] = name
    return msg


class RepeatGuardTests(unittest.TestCase):
    def test_rerun_after_edit_is_not_repeatable(self):
        messages = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "bash",
                                '{"command": "pytest -q"}')]},
            _tool_msg("c1", "1 failed", name="bash"),
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c2", "edit",
                                '{"path": "a.py"}')]},
            _tool_msg("c2", "ok", name="edit"),
        ]
        # The pytest result predates a mutating edit: state changed, so a
        # re-run with identical args is new work, not a loop.
        self.assertEqual(_repeatable_calls(messages), set())

    def test_reread_after_edit_is_not_repeatable(self):
        messages = [
            {"role": "user", "content": "fix a.py"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "read", '{"path": "a.py"}')]},
            _tool_msg("c1", "old", name="read"),
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c2", "edit", '{"path": "a.py"}')]},
            _tool_msg("c2", "ok", name="edit"),
        ]
        self.assertEqual(_repeatable_calls(messages), set())

    def test_pure_read_loop_stays_repeatable(self):
        messages = [
            {"role": "user", "content": "read a.py"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "read", '{"path": "a.py"}')]},
            _tool_msg("c1", "content", name="read"),
        ]
        self.assertIn(("read", '{"path": "a.py"}'),
                      _repeatable_calls(messages))

    def test_google_repeatable_cleared_by_mutation(self):
        contents = [
            {"parts": [{"functionCall": {"name": "read",
                                         "args": {"path": "a.py"}}}]},
            {"parts": [{"functionResponse": {"name": "read",
                                              "response": {"ok": 1}}}]},
            {"parts": [{"functionCall": {"name": "edit",
                                         "args": {"path": "a.py"}}}]},
            {"parts": [{"functionResponse": {"name": "edit",
                                              "response": {"ok": 1}}}]},
        ]
        self.assertEqual(_google_repeatable(contents), set())


class EvidenceCountingTests(unittest.TestCase):
    def test_glob_pattern_is_not_a_read(self):
        messages = [
            {"role": "user", "content": "review source"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "glob",
                                '{"pattern": "**/hand3d.py"}')]},
            _tool_msg("c1", "hand3d.py", name="glob"),
        ]
        self.assertEqual(files_read_in_turn(messages), set())

    def test_write_edit_bash_are_not_reads(self):
        for name, args in (
                ("write", '{"path": "x.py", "content": "hi"}'),
                ("edit", '{"path": "x.py"}'),
                ("bash", '{"command": "ls x.py"}')):
            with self.subTest(tool=name):
                messages = [
                    {"role": "user", "content": "do it"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [_tc("c1", name, args)]},
                    _tool_msg("c1", "ok", name=name),
                ]
                self.assertEqual(files_read_in_turn(messages), set())

    def test_reader_path_key_counts(self):
        messages = [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "read", '{"path": "server.py"}')]},
            _tool_msg("c1", "code", name="read"),
        ]
        self.assertEqual(files_read_in_turn(messages), {"server.py"})

    def test_long_file_body_is_not_a_listing(self):
        body = "code mentioning config.json and README.md\n" * 200
        messages = [
            {"role": "user", "content": "review source"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "read", '{"path": "a.py"}')]},
            _tool_msg("c1", body, name="read"),
        ]
        self.assertEqual(listed_turn_files(messages), [])

    def test_short_listing_still_counts(self):
        messages = [
            {"role": "user", "content": "review source"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "glob", '{"pattern": "**/*.py"}')]},
            _tool_msg("c1", "a.py\nb.py", name="glob"),
        ]
        self.assertEqual(sorted(listed_turn_files(messages)), ["a.py", "b.py"])

    def test_member_access_is_not_a_file(self):
        self.assertEqual(_files_in("self.cmd"), set())
        self.assertEqual(_files_in("process.env"), set())
        self.assertEqual(_files_in("math.log(x)"), set())
        self.assertEqual(_files_in("docs.rs"), set())
        self.assertEqual(_files_in("args.go"), set())

    def test_paths_and_quoted_names_count(self):
        self.assertIn("src/server.py", _files_in("see src/server.py"))
        self.assertIn("app.log", _files_in("open `app.log`"))
        self.assertIn("server.py", _files_in("read server.py"))


class NoHardcodedInstructionTests(unittest.TestCase):
    def test_no_foreign_filenames_in_server_source(self):
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
            "gemini_web2api", "server.py").read_text(encoding="utf-8")
        for leaked in ("hand3d.py", "vkeyboard.py", "scorereader.py",
                       "handmesh.py", "handspec.py"):
            self.assertNotIn(leaked, src)

    def test_review_guidance_lives_in_tool_block(self):
        tools = [{"type": "function", "function": {
            "name": "read", "description": "read",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}}]
        prompt, _ = messages_to_prompt([{"role": "user", "content": "hi"}],
                                       tools)
        self.assertIn("Listing or globbing file names is not reading", prompt)
        self.assertNotIn("hand3d", prompt)


class MissedToolVerbTests(unittest.TestCase):
    def test_implement_verbs_are_misses(self):
        for text in ("I'll create the file now.",
                     "Let me update the code.",
                     "Now I'll run the tests.",
                     "I will implement the plan.",
                     "De minh sua file nay.",
                     "Minh se chay test ngay."):
            with self.subTest(text=text):
                self.assertTrue(looks_like_missed_tool_call(text))

    def test_single_effective_threshold(self):
        self.assertEqual(_MAX_MISSED_TOOL_CHARS, 1200)


class PlanPinningTests(unittest.TestCase):
    def test_first_user_message_survives_truncation(self):
        plan = "PLAN-UNIQUE-TOKEN implement feature X step by step"
        middle = "filler middle content " * 20000
        parts = ["[System instruction]: be helpful", plan, middle,
                 "THE FINAL QUESTION"]
        prompt = _join_prompt_parts(parts, 60000)
        self.assertIn("PLAN-UNIQUE-TOKEN", prompt)
        self.assertIn("THE FINAL QUESTION", prompt)
        self.assertLessEqual(len(prompt.encode("utf-8")), 60000)

    def test_squeezed_tool_pin_keeps_constraint(self):
        from gemini_web2api.tools import _squeeze_part
        block = ("# Tool Use header\n" + "x" * 50000
                 + '\nIMPORTANT: You MUST call the tool "read".')
        squeezed = _squeeze_part(block, 60000 * 2 // 5)
        self.assertIn('MUST call the tool "read"', squeezed)


class CreationIntentTests(unittest.TestCase):
    def test_create_new_file_is_not_uncovered(self):
        self.assertIsNone(pending_tool_request(
            [{"role": "user",
              "content": "Create a new file report.txt with the summary."}]))

    def test_read_plan_then_implement_is_not_source_review(self):
        self.assertFalse(_is_source_review("doc ke hoach roi implement"))
        self.assertFalse(_is_source_review("read the plan then implement it"))
        self.assertTrue(_is_source_review("review the plan carefully"))
        self.assertTrue(_is_source_review("check the source code"))


class GoogleParityTests(unittest.TestCase):
    def test_undeclared_google_name_dropped(self):
        text = ('```function_call\n{"name": "bogus", "args": {}}\n```')
        clean, calls = parse_google_function_calls(text, allowed_names={"read"},
                                                   tool_schemas={"read": {"path"}})
        self.assertEqual(calls, [])
        self.assertIn("bogus", clean)

    def test_prefixed_google_name_normalised(self):
        text = ('```function_call\n{"name": "functions.read", '
                '"args": {"path": "a.py"}}\n```')
        _clean, calls = parse_google_function_calls(
            text, allowed_names={"read"}, tool_schemas={"read": {"path"}})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read")

    def test_placeholder_google_args_dropped(self):
        text = ('```function_call\n{"name": "read", '
                '"args": {"path": "EXAMPLE_path"}}\n```')
        clean, calls = parse_google_function_calls(
            text, allowed_names={"read"}, tool_schemas={"read": {"path"}})
        self.assertEqual(calls, [])
        self.assertIn("function_call", clean)


class ParseRobustnessTests(unittest.TestCase):
    def test_mid_sentence_fence_is_not_a_call(self):
        text = ('For example ```tool_call\n{"name": "read", '
                '"arguments": {"path": "a.py"}}\n``` in prose.')
        clean, calls, _dropped = parse_tool_calls(
            text, allowed_names={"read"}, tool_schemas={"read": {"path"}})
        self.assertEqual(calls, [])
        self.assertIn("tool_call", clean)

    def test_line_start_fence_is_a_call(self):
        text = ('```tool_call\n{"name": "read", '
                '"arguments": {"path": "a.py"}}\n```')
        _clean, calls, _dropped = parse_tool_calls(
            text, allowed_names={"read"}, tool_schemas={"read": {"path"}})
        self.assertEqual(len(calls), 1)

    def test_placeholder_args_dropped_with_reason(self):
        text = ('```tool_call\n{"name": "read", '
                '"arguments": {"path": "EXAMPLE_path"}}\n```')
        clean, calls, dropped = parse_tool_calls(
            text, allowed_names={"read"}, tool_schemas={"read": {"path"}})
        self.assertEqual(calls, [])
        self.assertTrue(any("EXAMPLE" in d for d in dropped))
        self.assertIn("tool_call", clean)

    def test_declared_string_input_param_kept(self):
        args = extract_arguments({"name": "summarise", "input": "hello"},
                                 props={"input"})
        self.assertEqual(args, {"input": "hello"})

    def test_wrapper_input_string_still_works(self):
        args = extract_arguments({"name": "read", "input": '{"path": "a"}'})
        self.assertEqual(args, '{"path": "a"}')

    def test_fenced_markers_survive_strip(self):
        text = ("Here is the format:\n```\n[Assistant]: hello\n```\nDone.")
        self.assertIn("[Assistant]: hello", _strip_fabricated(text, False))

    def test_real_fabrication_still_cut(self):
        text = "Answer done.\n[Assistant]: I will continue myself."
        self.assertNotIn("continue myself", _strip_fabricated(text, False))


class RequiredToolContractTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["tool_retry_on_miss"] = True

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def _retry(self, generate, messages, **kwargs):
        import threading as _t
        params = dict(prompt="p", model_id=2, think_mode=1, file_refs=None,
                      extra=None, tools_active=True, allowed_names={"read"},
                      tool_schemas={"read": {"path"}}, required_tool=True,
                      messages=messages, ticket=None,
                      tool_required={"read": {"path"}},
                      client_gone=_t.Event())
        params.update(kwargs)
        return GeminiHandler.__new__(GeminiHandler)._generate_with_tool_retry(
            **params)

    @mock.patch("gemini_web2api.server.generate")
    def test_required_prose_only_raises(self, generate):
        generate.side_effect = ["just prose", "more prose"]
        with self.assertRaises(GeminiError):
            self._retry(generate, [{"role": "user", "content": "read a.py"}])

    @mock.patch("gemini_web2api.server.generate")
    def test_required_duplicate_falls_back_to_prose(self, generate):
        CONFIG["tool_retry_attempts"] = 0
        history = [
            {"role": "user", "content": "read a.py"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "content": "data", "tool_call_id": "c1"},
        ]
        generate.side_effect = [
            '```tool_call\n{"name": "read", '
            '"arguments": {"path": "a.py"}}\n```']
        text, tool_calls = self._retry(generate, history)
        self.assertFalse(tool_calls)
        self.assertIsInstance(text, str)


class LogRedactionTests(unittest.TestCase):
    def test_tool_trace_redacts_long_values(self):
        import io
        from contextlib import redirect_stderr
        from gemini_web2api.server import _log_tool_trace
        secret = "SECRET-BODY-" + "x" * 5000
        calls = [{"id": "c1", "type": "function",
                  "function": {"name": "write",
                               "arguments": json.dumps({"path": "a.py",
                                                        "content": secret})}}]
        with mock.patch.dict(CONFIG, {"log_requests": True}):
            buf = io.StringIO()
            with redirect_stderr(buf):
                _log_tool_trace("```tool_call\n{}\n```", calls)
        out = buf.getvalue()
        self.assertNotIn(secret, out)
        self.assertIn("truncated", out)


class ChatEndpointRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
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
        connection = http.client.HTTPConnection("127.0.0.1", self.port,
                                                timeout=5)
        connection.request("POST", path, body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()
        return response.status, body

    @mock.patch("gemini_web2api.server.generate", return_value="ok answer")
    def test_no_injected_foreign_instruction(self, generate):
        status, _body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "user", "content": "review my code"}]})
        self.assertEqual(status, 200)
        self.assertNotIn("hand3d", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.generate", return_value="ok answer")
    def test_client_system_prompt_preserved(self, generate):
        status, _body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [{"role": "system", "content": "Talk like a pirate."},
                          {"role": "user", "content": "review my code"}]})
        self.assertEqual(status, 200)
        prompt = generate.call_args.args[0]
        self.assertIn("Talk like a pirate.", prompt)
        self.assertNotIn("hand3d", prompt)

    @mock.patch("gemini_web2api.server.generate", return_value="ok answer")
    def test_list_system_content_does_not_500(self, generate):
        status, _body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash",
             "messages": [
                 {"role": "system",
                  "content": [{"type": "text", "text": "Be brief."}]},
                 {"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)

    def test_empty_messages_still_400(self):
        status, _body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash", "messages": []})
        self.assertEqual(status, 400)

    @mock.patch("gemini_web2api.server.generate")
    def test_repeat_read_after_edit_allowed(self, generate):
        tools = [{"type": "function", "function": {
            "name": "read", "description": "r",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}},
                 {"type": "function", "function": {
                     "name": "edit", "description": "e",
                     "parameters": {"type": "object",
                                    "properties": {"path": {"type": "string"}}}}}]
        history = [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read",
                             "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "old"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c2", "type": "function",
                "function": {"name": "edit",
                             "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"},
            {"role": "user", "content": "verify the fix"},
        ]
        generate.return_value = (
            '```tool_call\n{"name": "read", '
            '"arguments": {"path": "a.py"}}\n```')
        status, body = self.post_json(
            "/v1/chat/completions",
            {"model": "gemini-3.6-flash", "messages": history,
             "tools": tools})
        self.assertEqual(status, 200)
        self.assertIn("tool_calls", body)


class VaguePromptEvidenceTests(unittest.TestCase):
    """Live-test findings: vague implement/review prompts must still seek evidence.

    Observed against a copy of this repo via opencode:
    - "Review how this proxy loads and caches the session cookie..." got a
      generic checklist with zero tool calls (no source noun matched).
    - "The health check should also work under /v1... Make it so" did
      glob+grep then described the edit instead of reading/editing.
    - "Find where the retry count is configured and double it" grepped once
      then pointed at the wrong setting without reading anything.
    """

    def test_review_of_unnamed_system_is_source_review(self):
        self.assertTrue(_is_source_review(
            "review how this proxy loads and caches the google session "
            "cookie, and flag any security or correctness issues"))

    def test_vague_review_prose_gets_read_retry(self):
        # `owed` is computed before the reply, so a vague first-turn review
        # ("review how this proxy...") can only be caught reply-side: prose
        # with no call ever emitted must cost one retry, not a 200.
        self.assertIsNone(pending_tool_request(
            [{"role": "user",
              "content": "Review how this proxy loads and caches the Google "
                         "session cookie, and flag any security or "
                         "correctness issues"}]))

    def test_status_check_stays_silent(self):
        self.assertFalse(_is_source_review("check the server status"))
        self.assertFalse(change_demanded(
            [{"role": "user", "content": "check the server status"}]))
        self.assertIsNone(pending_tool_request(
            [{"role": "user", "content": "check the server status"}]))

    def test_math_word_problem_stays_silent(self):
        self.assertFalse(change_demanded(
            [{"role": "user", "content": "double it and add five"}]))
        self.assertIsNone(pending_tool_request(
            [{"role": "user", "content": "double it and add five"}]))

    def test_make_it_so_is_change_demanded(self):
        self.assertTrue(change_demanded(
            [{"role": "user",
              "content": "The health check should also work under the /v1 "
                         "prefix. Make it so, and verify with the test "
                         "suite"}]))

    def test_survey_only_turn_still_owes_change(self):
        messages = [
            {"role": "user",
             "content": "The health check should also work under the /v1 "
                        "prefix. Make it so"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "glob", '{"pattern": "**/*"}')]},
            _tool_msg("c1", "server.py\nworker.js", name="glob"),
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c2", "grep", '{"pattern": "health"}')]},
            _tool_msg("c2", "server.py: health found", name="grep"),
        ]
        owed = pending_tool_request(messages)
        self.assertIsNotNone(owed)
        self.assertIn("change requested", owed)

    def test_grep_only_config_turn_still_owes_change(self):
        messages = [
            {"role": "user",
             "content": "Find where the tool-call retry count is configured "
                        "and double it"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "grep", '{"pattern": "retry"}')]},
            _tool_msg("c1", "100 matches", name="grep"),
        ]
        owed = pending_tool_request(messages)
        self.assertIsNotNone(owed)
        self.assertIn("change requested", owed)

    def test_read_then_edit_turn_owes_nothing(self):
        messages = [
            {"role": "user", "content": "Fix the crash in server.py"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "read", '{"path": "server.py"}')]},
            _tool_msg("c1", "code...", name="read"),
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c2", "edit", '{"path": "server.py"}')]},
            _tool_msg("c2", "ok", name="edit"),
        ]
        self.assertIsNone(pending_tool_request(messages))

    def test_prior_turn_read_satisfies_change(self):
        messages = [
            {"role": "user", "content": "Add the endpoint"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "read", '{"path": "server.py"}')]},
            _tool_msg("c1", "code...", name="read"),
            {"role": "user", "content": "now add it"},
        ]
        self.assertIsNone(pending_tool_request(messages))

    def test_creation_turn_owes_nothing(self):
        messages = [
            {"role": "user",
             "content": "Create primes.txt with the first five primes"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "write",
                                '{"path": "primes.txt"}')]},
            _tool_msg("c1", "ok", name="write"),
        ]
        self.assertIsNone(pending_tool_request(messages))

    def test_prompt_block_teaches_survey_leads_rule(self):
        prompt, _images = messages_to_prompt(
            [{"role": "user", "content": "hi"}],
            [{"type": "function",
              "function": {"name": "read", "description": "r",
                           "parameters": {"type": "object",
                                          "properties": {
                                              "path": {"type": "string"}}}}}])
        self.assertIn("leads, not evidence", prompt)
        self.assertIn("re-read the changed lines", prompt)

    def test_prompt_block_has_fixed_flow(self):
        prompt, _images = messages_to_prompt(
            [{"role": "user", "content": "hi"}],
            [{"type": "function",
              "function": {"name": "read", "description": "r",
                           "parameters": {"type": "object",
                                          "properties": {
                                              "path": {"type": "string"}}}}}])
        self.assertIn("Fixed flow for EVERY turn", prompt)
        self.assertIn("are never an answer", prompt)

    @mock.patch("gemini_web2api.server.generate")
    def test_vague_review_prose_retried_once(self, generate):
        tools = [{"type": "function", "function": {
            "name": "read", "description": "r",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}}]
        generate.side_effect = [
            "A generic checklist with no tool call.",
            '```tool_call\n{"name": "read", '
            '"arguments": {"path": "gemini.py"}}\n```',
        ]
        server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            original = dict(CONFIG)
            CONFIG["api_keys"] = []
            CONFIG["log_requests"] = False
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5)
            connection.request(
                "POST", "/v1/chat/completions",
                body=json.dumps(
                    {"model": "gemini-3.6-flash",
                     "messages": [{"role": "user",
                                   "content": "Review how this proxy loads "
                                              "and caches the session "
                                              "cookie"}],
                     "tools": tools}),
                headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read().decode()
            connection.close()
            CONFIG.clear()
            CONFIG.update(original)
            self.assertEqual(response.status, 200)
            self.assertEqual(generate.call_count, 2)
            self.assertIn("tool_calls", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @mock.patch("gemini_web2api.server.generate")
    def test_survey_stall_prose_retried_once(self, generate):
        tools = [{"type": "function", "function": {
            "name": "grep", "description": "g",
            "parameters": {"type": "object",
                           "properties": {"pattern": {"type": "string"}}}}},
            {"type": "function", "function": {
                "name": "read", "description": "r",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}}}}}]
        history = [
            {"role": "user",
             "content": "Find where the retry count is configured and "
                        "double it"},
            {"role": "assistant", "content": "",
             "tool_calls": [_tc("c1", "grep", '{"pattern": "retry"}')]},
            _tool_msg("c1", "100 matches", name="grep"),
        ]
        generate.side_effect = [
            "Edit config.example.json and change 3 to 6.",
            '```tool_call\n{"name": "read", '
            '"arguments": {"path": "config.py"}}\n```',
        ]
        server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            original = dict(CONFIG)
            CONFIG["api_keys"] = []
            CONFIG["log_requests"] = False
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5)
            connection.request(
                "POST", "/v1/chat/completions",
                body=json.dumps({"model": "gemini-3.6-flash",
                                 "messages": history, "tools": tools}),
                headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read().decode()
            connection.close()
            CONFIG.clear()
            CONFIG.update(original)
            self.assertEqual(response.status, 200)
            self.assertEqual(generate.call_count, 2)
            self.assertIn("tool_calls", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
