"""Live-upstream plumbing: BL label, cookie export sync, BardError detection.

These cover the failure modes seen against the real gemini.google.com:
- the build label family was renamed (boq_assistant-bard-web-server ->
  boq_gemini-web-uiserver), which silently disabled BL auto-update;
- the extension export carries xsrf/auth fields that must reach CONFIG or the
  first request goes out without `at` and Google answers HTTP 400;
- rejections now arrive as protobuf-JSON (BardErrorInfo",[1155]]) which the
  legacy prose regex did not match, turning a rejection into "empty response".
"""
import json
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from gemini_web2api import gemini
from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import (
    GeminiError,
    _build_payload,
    _status_error,
    bard_error_code,
    extract_response_text,
    fetch_latest_bl,
    load_cookie,
    raise_bard_error,
)

LEGACY_LABEL = "boq_assistant-bard-web-server_20260716.08_p0"
CURRENT_LABEL = "boq_gemini-web-uiserver_20261007.12_p0"

# The shape Google actually sends today (plus the legacy prose form).
PROTOBUF_ERROR = (
    ')]}\'\n159\n[["wrb.fr",null,"[null,[null,\\"r_abc\\"],{}]\"]]\n122\n'
    '[["wrb.fr",null,null,null,null,[13,null,'
    '[["type.googleapis.com/assistant.boq.bard.application.BardErrorInfo",'
    "[1155]]]]]]\n"
)
PROSE_ERROR = 'something\n[["wrb.fr",null,"... BardErrorInfo [1099] ..."]]\n'


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body


class FetchLatestBlTests(unittest.TestCase):
    """The label pattern must survive Google renaming the boq_* family."""

    def _fetch(self, html: str):
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse(html.encode())):
            return fetch_latest_bl()

    def test_parses_current_uiserver_label(self):
        html = '<script>var a={"cfb2h":"%s"};</script>' % CURRENT_LABEL
        self.assertEqual(self._fetch(html), CURRENT_LABEL)

    def test_parses_legacy_bard_label(self):
        html = 'x boq_assistant-bard-web-server_20260716.08_p0 y'
        self.assertEqual(self._fetch(html), LEGACY_LABEL)

    def test_ignores_bundle_urls_without_a_build_label(self):
        # Script URLs contain "boq-<family>.<Module>" fragments that must not
        # be mistaken for a label.
        html = 'k=boq-gemini-web-uiserver.BardChatUi.en_US.2018.O/am=x'
        self.assertIsNone(self._fetch(html))

    def test_returns_none_when_absent(self):
        self.assertIsNone(self._fetch("<html>no label here</html>"))


class CookieExportSyncTests(unittest.TestCase):
    """cookie_file alone must be enough: the export carries the auth fields."""

    def setUp(self):
        self._saved = {k: CONFIG.get(k) for k in
                       ("cookie_file", "xsrf_token", "auth_user", "gemini_bl")}
        self._cache = gemini._cookie_cache
        gemini._cookie_cache = ("", None, 0)
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

    def tearDown(self):
        CONFIG.update(self._saved)
        gemini._cookie_cache = self._cache
        os.unlink(self.path)

    def test_export_fields_reach_config(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"cookie": "SSID=x; SAPISID=y", "sapisid": "y",
                       "auth_user": 2, "xsrf_token": "TOKEN:123",
                       "gemini_bl": CURRENT_LABEL}, fh)
        CONFIG.update({"cookie_file": self.path, "xsrf_token": None,
                       "auth_user": None, "gemini_bl": LEGACY_LABEL})

        cookie, sapisid = load_cookie()

        self.assertIn("SSID=x", cookie)
        self.assertEqual(sapisid, "y")
        self.assertEqual(CONFIG["xsrf_token"], "TOKEN:123")
        self.assertEqual(CONFIG["auth_user"], 2)
        self.assertEqual(CONFIG["gemini_bl"], CURRENT_LABEL)

    def test_payload_carries_at_from_the_cookie_file(self):
        # _build_payload runs before any header is built; if it does not load
        # the cookie itself, the first attempt goes out with no `at` field and
        # Google answers 400 (which is treated as non-retryable).
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"cookie": "SSID=x", "xsrf_token": "TOKEN:123"}, fh)
        CONFIG.update({"cookie_file": self.path, "xsrf_token": None})

        body = _build_payload("hi", 1, 4)

        parsed = urllib.parse.parse_qs(body)
        self.assertEqual(parsed.get("at"), ["TOKEN:123"])

    def test_plain_cookie_txt_does_not_clobber_config(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("SSID=x; SAPISID=y")
        CONFIG.update({"cookie_file": self.path, "xsrf_token": "KEEP",
                       "auth_user": None, "gemini_bl": LEGACY_LABEL})

        load_cookie()

        self.assertEqual(CONFIG["xsrf_token"], "KEEP")
        self.assertEqual(CONFIG["gemini_bl"], LEGACY_LABEL)


class BardErrorDetectionTests(unittest.TestCase):
    def test_detects_protobuf_json_form(self):
        self.assertEqual(bard_error_code(PROTOBUF_ERROR), 1155)

    def test_detects_legacy_prose_form(self):
        self.assertEqual(bard_error_code(PROSE_ERROR), 1099)

    def test_no_error_on_normal_payload(self):
        self.assertIsNone(bard_error_code('["wrb.fr",null,"[null,[null,\\"r_x\\"],{}]"]'))
        # No exception: a healthy payload must pass through untouched.
        raise_bard_error("plain answer, no rejection in here")
        self.assertIsNone(bard_error_code("plain answer"))

    def test_raises_gemini_error_with_code(self):
        with self.assertRaises(GeminiError) as ctx:
            raise_bard_error(PROTOBUF_ERROR)
        self.assertIn("BardErrorInfo [1155]", str(ctx.exception))
        # 1155 is not in the hint table: no dangling suffix.
        self.assertNotIn("--", str(ctx.exception))

    def test_known_code_carries_hint(self):
        raw = PROSE_ERROR.replace("1099", "1037")
        with self.assertRaises(GeminiError) as ctx:
            raise_bard_error(raw)
        self.assertIn("BardErrorInfo [1037]", str(ctx.exception))
        self.assertIn("usage limit exceeded", str(ctx.exception))

    def test_extract_raises_instead_of_returning_empty(self):
        # Before the fix this returned "" and the client saw a misleading
        # "empty response from upstream".
        with self.assertRaises(GeminiError) as ctx:
            extract_response_text(PROTOBUF_ERROR)
        self.assertIn("1155", str(ctx.exception))


class StatusErrorTests(unittest.TestCase):
    def setUp(self):
        self._saved = CONFIG.get("cookie_file")

    def tearDown(self):
        CONFIG["cookie_file"] = self._saved

    def test_400_without_cookie_explains_anonymous_rejection(self):
        CONFIG["cookie_file"] = None
        err = _status_error(400)
        self.assertEqual(err.status, 400)
        self.assertIn("cookie_file", err.args[0])

    def test_400_with_cookie_stays_terse(self):
        CONFIG["cookie_file"] = "cookie.txt"
        err = _status_error(400)
        self.assertEqual(err.status, 400)
        self.assertNotIn("cookie_file", err.args[0])

    def test_other_statuses_have_no_cookie_suffix(self):
        CONFIG["cookie_file"] = None
        self.assertNotIn("cookie_file", _status_error(429).args[0])


if __name__ == "__main__":
    unittest.main()
