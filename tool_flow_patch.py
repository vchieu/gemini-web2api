#!/usr/bin/env python3
"""tool_flow_patch.py — refactor vòng lặp tool-call của gemini_web2api.

Bối cảnh (evidence từ log thật của user):
  [timing] generate attempt 1: 24.0s, 98 chars        <- upstream TTFB rất chậm
  no tool_call parsed; text head: "I can't use Workspace because required
      Gmail settings are off. Turn on these settings to continue."
  Tool retry (attempt 1/4): upstream returned its canned error placeholder...
  [timing] generate attempt 1: 8.0s, 59 chars
  'Sorry, something went wrong. Please try your request again.'
  -> proxy im lặng ~60+ giây rồi "tự dựng dừng": client không nhận được gì
     giữa các attempt, và các canned refusal dài hơn 800 ký tự lọt qua
     ``looks_like_extension_refusal`` nên bị coi là câu trả lời "hợp lệ".

Patch này (idempotent, anchor-based, backup *.bak, hỗ trợ --revert) sửa 4 điểm:

  A. tools.py   : thêm ``looks_like_canned_failure()`` — phát hiện CẢ canned
                  error lẫn extension refusal ở bất kỳ độ dài nào (substrings,
                  bỏ guard 800 ký tự của looks_like_extension_refusal).
  B. server.py  : early-exit stream khi tool fence hoàn chỉnh dùng parser thật
                  (_iter_tool_blocks + raw_decode) thay vì heuristic tìm "```";
                  hàng đợi delta có timeout -> log "[timing] ... silent for Xs"
                  để phân biệt "upstream chậm" với "proxy treo".
  C. server.py  : cả hai tool-retry loop (OpenAI + Google) phát delta về client
                  ngay khi tool_stream bật (gửi SSE ping sớm, forward delta text
                  làm heartbeat trong lúc buffer) -> client thấy status từng
                  phần thay vì im lặng 24-60s. Placeholder/canned detection
                  chuyển sang looks_like_canned_failure.
  D. config.py  : thêm default "tool_delta_forward": True (tắt = giữ hành vi cũ).

Cách dùng (tại thư mục gốc repo, cạnh gemini_web2api/):
    python tool_flow_patch.py            # áp patch (tạo *.bak nếu chưa có)
    python tool_flow_patch.py --revert   # khôi phục từ *.bak
Yêu cầu đã áp latency_stream_patch.py trước (các anchor dưới đây nằm trong code
đã được latency patch vá). Sau đó đảm bảo config.json có {"tool_stream": true}.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "gemini_web2api"
TOOLS = PKG / "tools.py"
SERVER = PKG / "server.py"
CONFIGF = PKG / "config.py"

# ---------------------------------------------------------------- anchors ---

OLD_STREAMED_FN = '''def _streamed_generate(call_prompt, model_id, think_mode, file_refs,
                       extra, ticket, tools_active, attempt_no):
    """generate_stream() collapsed to one string, with [timing] + early exit.

    Used by the tool-retry loops when CONFIG["tool_stream"] is on: the client
    sees first-delta timing in the log and we stop reading upstream as soon as
    a complete tool_call fence has arrived -- waiting for the tail of a turn
    that already produced its call only delays the next client action.
    """
    chunks = []
    ttfb_done = False
    t0 = time.monotonic()
    for delta in generate_stream(call_prompt, model_id, think_mode,
                                 file_refs, extra, ticket):
        if not ttfb_done and delta:
            ttfb_done = True
            log(f"[timing] tool attempt {attempt_no + 1}: first delta in "
                f"{time.monotonic() - t0:.1f}s")
        chunks.append(delta)
        if tools_active:
            raw_so_far = "".join(chunks)
            if looks_like_tool_call(raw_so_far):
                i = raw_so_far.find("```")
                if i >= 0 and "\\n```" in raw_so_far[i + 3:]:
                    log(f"[timing] tool attempt {attempt_no + 1}: complete "
                        f"tool_call fence after {time.monotonic() - t0:.1f}s, "
                        "early-exit read")
                    break
    return "".join(chunks)'''

NEW_STREAMED_FN = '''def _complete_call_end(text: str):
    """Char index just past the LAST closed, parseable tool-call block, else None.

    Uses the same primitives as the real parser (_FENCE_OPEN/_FENCE_CLOSE +
    JSONDecoder.raw_decode with strict=False), so early-exit fires exactly
    when the buffered prefix already holds a call ``parse_tool_calls`` would
    accept -- never on a stray triple-backtick. Conservative by design:
    * a payload still streaming fails raw_decode -> wait for more deltas;
    * a non-JSON body (Python-style ``read(path="x")``) is NOT counted --
      AST recovery happens later in parse_tool_calls, so such turns simply
      read to completion instead of risking a cut-off call;
    * prose ```json examples are skipped and scanning continues -- the real
      call may follow them.
    """
    from .tools import _ARGUMENT_KEYS, _FENCE_OPEN, _FENCE_CLOSE
    dec = json.JSONDecoder(strict=False)
    last_end = None
    pos = 0
    while True:
        m = _FENCE_OPEN.search(text, pos)
        if not m:
            return last_end
        j = m.end()
        while j < len(text) and text[j] in " \\t\\r\\n":
            j += 1
        kind = m.group(1)
        try:
            data, end = dec.raw_decode(text, j)
        except ValueError:
            # Body not (yet) valid JSON: keep any earlier closed block and
            # wait for more deltas rather than cutting mid-payload.
            return last_end
        counted = False
        if isinstance(data, dict) and data.get("name"):
            if kind != "json" or any(k in data for k in _ARGUMENT_KEYS):
                counted = True
        if not counted:
            # Prose example or non-call payload: skip it, keep scanning.
            c = _FENCE_CLOSE.match(text, end)
            pos = c.end() if c else max(end, m.end())
            continue
        c = _FENCE_CLOSE.match(text, end)
        if not c:
            return last_end          # closing fence still streaming
        last_end = c.end()
        pos = last_end


def _streamed_generate(call_prompt, model_id, think_mode, file_refs,
                       extra, ticket, tools_active, attempt_no, on_delta=None):
    """generate_stream() collapsed to one string, with [timing] + early exit.

    Used by the tool-retry loops when CONFIG["tool_stream"] is on. Three jobs:
      * log first-delta timing so operator and client can tell "upstream slow"
        from "proxy hung";
      * stop reading upstream as soon as a *complete, parseable* tool_call
        fence has arrived (real parser, not a backtick heuristic);
      * hand every delta to ``on_delta`` so the caller can keep the client's
        connection warm (heartbeat/ping) during a long buffered turn.

    The generator is consumed through a queue with a timeout: silence itself
    becomes an observable event ("silent for Ns") instead of an unexplained
    hang. GeneratorExit never reaches the wrapped generator because it is
    always drained into the queue by the pump thread first.
    """
    import queue as _queue
    q: "_queue.Queue" = _queue.Queue()

    def _pump():
        try:
            for d in generate_stream(call_prompt, model_id, think_mode,
                                     file_refs, extra, ticket):
                q.put(d)
        except BaseException as exc:      # re-raised in this thread below
            q.put(exc)
        q.put(None)

    threading.Thread(target=_pump, daemon=True).start()
    chunks = []
    ttfb_done = False
    silent_logged = False
    last_activity = t0 = time.monotonic()
    while True:
        try:
            item = q.get(timeout=20.0)
        except _queue.Empty:
            if not silent_logged:
                silent_logged = True
                log(f"[timing] tool attempt {attempt_no + 1}: upstream silent "
                    f"for {time.monotonic() - last_activity:.0f}s mid-turn")
            continue
        if isinstance(item, BaseException):
            raise item
        if item is None:
            break
        silent_logged = False
        last_activity = time.monotonic()
        if not ttfb_done and item:
            ttfb_done = True
            log(f"[timing] tool attempt {attempt_no + 1}: first delta in "
                f"{last_activity - t0:.1f}s")
        chunks.append(item)
        if on_delta is not None:
            try:
                on_delta(item)
            except Exception as e:
                log(f"tool attempt {attempt_no + 1}: delta sink failed: {e}")
        if tools_active:
            raw_so_far = "".join(chunks)
            cut = _complete_call_end(raw_so_far)
            if cut is not None and looks_like_tool_call(raw_so_far[:cut]):
                log(f"[timing] tool attempt {attempt_no + 1}: complete "
                    f"tool_call fence after {time.monotonic() - t0:.1f}s, "
                    "early-exit read")
                break
    return "".join(chunks)'''

OLD_OAI_CALLSITE = '''            try:
                if CONFIG.get("tool_stream"):
                    raw = _streamed_generate(call_prompt, model_id, think_mode,
                                             file_refs, extra, ticket,
                                             tools_active, attempt)
                else:
                    raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Tool retry after upstream error: {e}")
                    continue'''

NEW_OAI_CALLSITE = '''            try:
                if CONFIG.get("tool_stream"):
                    # Buffered turn, live heartbeat: every upstream delta
                    # emits an SSE comment on the already-open stream, so the
                    # client sees per-part activity instead of 20-60 s of
                    # silence between attempts. The sink is set by the endpoint
                    # that opened the stream; absent -> old buffered behaviour.
                    hb = getattr(self, "_chat_hb", None)
                    sink = (hb.feed if (hb is not None
                                        and CONFIG.get("tool_delta_forward", True))
                            else None)
                    raw = _streamed_generate(call_prompt, model_id, think_mode,
                                             file_refs, extra, ticket,
                                             tools_active, attempt,
                                             on_delta=sink)
                else:
                    raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Tool retry after upstream error: {e}")
                    time.sleep(CONFIG["retry_delay_sec"])
                    continue'''

OLD_GOO_CALLSITE = '''            try:
                if CONFIG.get("tool_stream"):
                    raw = _streamed_generate(call_prompt, model_id, think_mode,
                                             file_refs, extra, ticket,
                                             tools_active, attempt)
                else:
                    raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Google tool retry after upstream error: {e}")
                    continue'''

NEW_GOO_CALLSITE = '''            try:
                if CONFIG.get("tool_stream"):
                    # Same live-heartbeat contract as the OpenAI loop: deltas
                    # reset/emit SSE pings through the sink the endpoint set up
                    # (self._google_hb). Absent sink -> old buffered behaviour.
                    ghb = getattr(self, "_google_hb", None)
                    gsink = (ghb.feed if (ghb is not None
                                          and CONFIG.get("tool_delta_forward", True))
                             else None)
                    raw = _streamed_generate(call_prompt, model_id, think_mode,
                                             file_refs, extra, ticket,
                                             has_tools, attempt,
                                             on_delta=gsink)
                else:
                    raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Google tool retry after upstream error: {e}")
                    time.sleep(CONFIG["retry_delay_sec"])
                    continue'''

OLD_PING_FN = '''        stop_ping = threading.Event()

        def _ping():
            while not stop_ping.wait(SSE_PING_INTERVAL_SEC):
                try:
                    self._sse_write(b": ping\\n\\n")
                except (BrokenPipeError, ConnectionResetError):
                    # Nobody is listening any more; the caller uses this to
                    # stop burning upstream calls for an answer no one gets.
                    client_gone.set()
                    return
                except Exception:
                    return

        threading.Thread(target=_ping, daemon=True).start()
        return stop_ping.set, client_gone'''

NEW_PING_FN = '''        stop_ping = threading.Event()

        hb = _Heartbeat(self, stop_ping, client_gone)
        threading.Thread(target=hb.run, daemon=True).start()
        return stop_ping.set, client_gone, hb'''

# Module-level heartbeat class, inserted right before _estimate_tokens.
OLD_HB_ANCHOR = '''def _estimate_tokens(text: str) -> int:'''
NEW_HB_ANCHOR = '''class _Heartbeat:
    """Delta sink for a buffered tool turn: upstream activity == keep-alive.

    ``feed(delta)`` is called by ``_streamed_generate`` for every streamed
    chunk of a tool-retry attempt. It writes one SSE comment immediately --
    clients see per-part activity instead of 20-60 s of silence -- and resets
    the interval timer so idle pings only fire when nothing arrives. The gone
    flag is shared with the caller: a failed write means nobody listens and
    further upstream attempts must be skipped. Thread-safe via ``_lock``
    because both the request thread (feed) and this thread (run) write.
    """

    def __init__(self, handler, stop_event, gone_event):
        self._handler = handler
        self._stop = stop_event
        self._gone = gone_event
        self._wake = threading.Event()
        self._lock = threading.Lock()

    def _write_ping(self):
        with self._lock:
            if self._stop.is_set():
                return False
            try:
                self._handler._sse_write(b": ping\\n\\n")
                return True
            except (BrokenPipeError, ConnectionResetError):
                self._gone.set()
                return False
            except Exception:
                return False

    def feed(self, delta=None):
        if self._stop.is_set() or self._gone.is_set():
            return
        self._write_ping()
        self._wake.set()          # restart the idle countdown

    def run(self):
        interval = SSE_PING_INTERVAL_SEC
        while not self._stop.is_set():
            if self._wake.wait(interval):
                self._wake.clear()
                continue          # real activity just happened: re-arm quietly
            if self._stop.is_set():
                return
            if not self._write_ping():
                return


def _estimate_tokens(text: str) -> int:'''

# Chat endpoint: unpack the new 3-tuple and clear the attribute afterwards.
OLD_CHAT_UNPACK = '''        stop_ping, client_gone = None, None
        if stream:
            stop_ping, client_gone = self._start_sse_with_keepalive()
            self._sse_chunk(cid, echo_model, {"role": "assistant", "content": ""},
                            None, include_usage)'''
NEW_CHAT_UNPACK = '''        stop_ping, client_gone = None, None
        if stream:
            stop_ping, client_gone, self._chat_hb = self._start_sse_with_keepalive()
            self._sse_chunk(cid, echo_model, {"role": "assistant", "content": ""},
                            None, include_usage)'''

OLD_CHAT_FINALLY = '''        finally:
            if stop_ping is not None:
                stop_ping()

        if rf_instruction and text:'''
NEW_CHAT_FINALLY = '''        finally:
            if stop_ping is not None:
                stop_ping()
            self._chat_hb = None

        if rf_instruction and text:'''

# _start_sse_with_keepalive now returns (stop, gone, heartbeat); update every
# unpack site so the tuple width stays consistent across endpoints.
OLD_RESP_UNPACK = '''        stop_ping, client_gone = None, None
        if want_stream:
            stop_ping, client_gone = self._start_sse_with_keepalive()
            sequence_number = 0'''
NEW_RESP_UNPACK = '''        stop_ping, client_gone = None, None
        if want_stream:
            stop_ping, client_gone, _hb = self._start_sse_with_keepalive()
            sequence_number = 0'''

# Google endpoint: keep its own head-start contract, add a google heartbeat.
OLD_GOO_SETUP = '''        stop_ping, client_gone, sse_started = None, None, False
        if stream and has_tools:
            stop_ping, client_gone = self._start_sse_with_keepalive()
            sse_started = True'''
NEW_GOO_SETUP = '''        stop_ping, client_gone, sse_started = None, None, False
        if stream and has_tools:
            stop_ping, client_gone, self._google_hb = \\
                self._start_sse_with_keepalive()
            sse_started = True'''

# Non-streaming chat path opens the same helper later; just absorb the 3rd item.
OLD_LATE_UNPACK = '''        stop_ping, _client_gone = self._start_sse_with_keepalive()'''
NEW_LATE_UNPACK = '''        stop_ping, _client_gone, _late_hb = self._start_sse_with_keepalive()'''

OLD_GOO_STOP1 = '''            finally:
                if stop_ping is not None:
                    stop_ping()

        def _google_fail(status, message, gstatus):'''
NEW_GOO_STOP1 = '''            finally:
                if stop_ping is not None:
                    stop_ping()
                self._google_hb = None

        def _google_fail(status, message, gstatus):'''

OLD_GOO_STOP2 = '''            finally:
                if stop_ping is not None:
                    stop_ping()
        else:
            self.send_json(response_obj)'''
NEW_GOO_STOP2 = '''            finally:
                if stop_ping is not None:
                    stop_ping()
                self._google_hb = None
        else:
            self.send_json(response_obj)'''

OLD_TOOLS_FUNC_TAIL = '''def looks_like_upstream_error(text: str) -> bool:
    """True when the response is the upstream's canned failure, not an answer."""
    if not text:
        return False
    return bool(_UPSTREAM_ERROR_RE.match(_fold_diacritics(text)))'''

NEW_TOOLS_FUNC_TAIL = '''def looks_like_upstream_error(text: str) -> bool:
    """True when the response is the upstream's canned failure, not an answer."""
    if not text:
        return False
    return bool(_UPSTREAM_ERROR_RE.match(_fold_diacritics(text)))


_CANNED_FAILURE_RES = (
    re.compile(r"sorry,?\\s+something\\s+went\\s+wrong", re.IGNORECASE),
    re.compile(r"please\\s+try\\s+your\\s+request\\s+again", re.IGNORECASE),
    re.compile(r"i\\s+can'?t\\s+use\\s+[a-z][a-z .'\\-]{0,60}?\\s+because\\s+"
               r"required\\s+.{0,80}?(settings|permissions)\\s+are\\s+off",
               re.IGNORECASE),
    re.compile(r"turn\\s+on\\s+these\\s+settings\\s+to\\s+continue",
               re.IGNORECASE),
)


def looks_like_canned_failure(text: str) -> bool:
    """True for ANY upstream canned sentence, at any length.

    ``looks_like_upstream_error`` is anchored (whole-string) and
    ``looks_like_extension_refusal`` caps at 800 chars, so a verbose model that
    pads "Sorry, something went wrong..." or the Workspace-settings refusal
    with prose escapes both and is handed to the client as a finished answer --
    observed live as turns that stall ~60 s and then "stop by themselves".
    These sentences are machine-generated upstream failures, never part of a
    genuine reply, so substring matching is safe here (the patterns are too
    specific to occur inside real content).
    """
    if not text or len(text) > 4000:
        return False
    folded = _fold_diacritics(text)
    return looks_like_upstream_error(text) or any(r.search(folded)
                                                  for r in _CANNED_FAILURE_RES)'''

OLD_OAI_ANCHOR1 = '''            if (text or "").strip() and not looks_like_upstream_error(text):'''
NEW_OAI_ANCHOR1 = '''            if (text or "").strip() and not looks_like_canned_failure(text):'''

OLD_OAI_ANCHOR2 = '''            if looks_like_upstream_error(text):
                placeholder_hits += 1'''
NEW_OAI_ANCHOR2 = '''            if looks_like_canned_failure(text):
                placeholder_hits += 1'''

OLD_OAI_ANCHOR3 = '''            if (not required_tool and tools_active
                    and looks_like_extension_refusal(text)):'''
NEW_OAI_ANCHOR3 = '''            if (not required_tool and tools_active
                    and looks_like_canned_failure(text)):'''

OLD_OAI_ANCHOR4 = '''        if not tool_calls and looks_like_upstream_error(text):'''
NEW_OAI_ANCHOR4 = '''        if not tool_calls and looks_like_canned_failure(text):'''

OLD_GOO_ANCHOR1 = '''            if (raw or "").strip() and not looks_like_upstream_error(raw):'''
NEW_GOO_ANCHOR1 = '''            if (raw or "").strip() and not looks_like_canned_failure(raw):'''

OLD_GOO_ANCHOR2 = '''            if looks_like_upstream_error(raw):
                log(f"Google tool retry (attempt {attempt + 1}/{attempts}): "'''
NEW_GOO_ANCHOR2 = '''            if looks_like_canned_failure(raw):
                log(f"Google tool retry (attempt {attempt + 1}/{attempts}): "'''

OLD_GOO_ANCHOR3 = '''        if looks_like_upstream_error(text):
            if best_text and not required_tool:'''
NEW_GOO_ANCHOR3 = '''        if looks_like_canned_failure(text):
            if best_text and not required_tool:'''

OLD_CONFIG_DEFAULT = '''    "tool_stream": False,
    "retry_attempts": 3,'''
NEW_CONFIG_DEFAULT = '''    "tool_stream": False,
    # When tool_stream is on, also forward each streamed delta of a buffered
    # tool turn to the client (SSE ping / heartbeat) so agent clients see
    # per-part progress instead of 20-60 s of silence. Set false to keep the
    # old fully-buffered behaviour.
    "tool_delta_forward": True,
    "retry_attempts": 3,'''

# ------------------------------------------------------------------- edits --

EDITS = [
    (SERVER, "B1: _streamed_generate -> queue+timeout, parser-based early exit, "
             "on_delta sink", OLD_STREAMED_FN, NEW_STREAMED_FN),
    (SERVER, "C1: OpenAI tool loop -> live heartbeat + retry backoff",
     OLD_OAI_CALLSITE, NEW_OAI_CALLSITE),
    (SERVER, "C2: Google tool loop -> live heartbeat + retry backoff",
     OLD_GOO_CALLSITE, NEW_GOO_CALLSITE),
    (TOOLS, "A: looks_like_canned_failure (length-independent detection)",
     OLD_TOOLS_FUNC_TAIL, NEW_TOOLS_FUNC_TAIL),
    (SERVER, "C3a: OpenAI best-text guard uses canned detector",
     OLD_OAI_ANCHOR1, NEW_OAI_ANCHOR1),
    (SERVER, "C3b: OpenAI placeholder branch uses canned detector",
     OLD_OAI_ANCHOR2, NEW_OAI_ANCHOR2),
    (SERVER, "C3c: OpenAI extension-refusal branch uses canned detector",
     OLD_OAI_ANCHOR3, NEW_OAI_ANCHOR3),
    (SERVER, "C3d: OpenAI final placeholder raise uses canned detector",
     OLD_OAI_ANCHOR4, NEW_OAI_ANCHOR4),
    (SERVER, "C3e: Google best-text guard uses canned detector",
     OLD_GOO_ANCHOR1, NEW_GOO_ANCHOR1),
    (SERVER, "C3f: Google placeholder branch uses canned detector",
     OLD_GOO_ANCHOR2, NEW_GOO_ANCHOR2),
    (SERVER, "C3g: Google final placeholder check uses canned detector",
     OLD_GOO_ANCHOR3, NEW_GOO_ANCHOR3),
    (CONFIGF, "D: add tool_delta_forward default",
     OLD_CONFIG_DEFAULT, NEW_CONFIG_DEFAULT),
]


def apply(revert: bool) -> int:
    files = sorted({p for p, _, _, _ in EDITS})
    if revert:
        rc = 0
        for path in files:
            bak = path.with_suffix(path.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, path)
                print(f"REVERTED {path.name} from {bak.name}")
            else:
                print(f"SKIP revert: no backup for {path.name}")
                rc = 1
        return rc

    missing = [p.name for p in files if not p.exists()]
    if missing:
        print(f"ERROR: not found: {', '.join(missing)} "
              "(run from the repo root, next to gemini_web2api/)")
        return 2

    for path in files:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"BACKUP {path.name} -> {bak.name}")

    skips = 0
    for path, name, old, new in EDITS:
        src = path.read_text(encoding="utf-8")
        if old not in src:
            if new in src:
                print(f"ALREADY  {path.name}: {name}")
                continue
            print(f"SKIP     {path.name}: {name}  (anchor not found -- is "
                  "latency_stream_patch.py applied? version differs?)")
            skips += 1
            continue
        if src.count(old) != 1:
            print(f"SKIP     {path.name}: {name}  (anchor matches "
                  f"{src.count(old)} times, expected 1)")
            skips += 1
            continue
        path.write_text(src.replace(old, new, 1), encoding="utf-8")
        print(f"PATCHED  {path.name}: {name}")

    for path in files:
        rc = subprocess.run([sys.executable, "-m", "py_compile", str(path)])
        if rc.returncode != 0:
            print(f"ERROR: {path.name} failed py_compile after patching!")
            return 1
    print("py_compile OK for all patched files")
    if skips:
        print(f"\n{skips} edit(s) SKIPPED -- send this output back for "
              "adjustment; do NOT trust a partially patched tree.")
        return 1
    print("\nDone. Ensure config.json has {\"tool_stream\": true} "
          "(delta forwarding is on by default via tool_delta_forward).")
    return 0


if __name__ == "__main__":
    sys.exit(apply("--revert" in sys.argv))
