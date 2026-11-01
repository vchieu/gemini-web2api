#!/usr/bin/env python3
"""tool_flow_final_patch.py (v2) -- patch "cuoi cung" cho loi
test_stream_attempt_exits_early_on_complete_fence ('TAIL SHOULD NOT BE READ').

CHAN DOAN (bang evidence, khong suy doan):
  Loi nay xay ra DUY NHAT khi _streamed_generate dang chay o bien the
  PUMP-THREAD (queue + worker thread): worker pull delta TAIL vao queue
  TRUOC khi vong lap kip break -> test thay 'TAIL SHOULD NOT BE READ'.
  Ban patch cu cua toi idempotent khong hoan toan: chay 2 lan thi dinh
  nghia bi trung lap va ban nam CUOI file moi duoc Python bind. Neu ban
  pump-thread nam cuoi -> loi he thong nhu ban bao cao.

Patch nay (idempotent, co dedupe, co --revert, tu kiem tra = in ket qua test):
  A. server.py : loai moi ban SAO CHEP cua _complete_call_end /
     _streamed_generate / _Heartbeat, GIU LAI DUY NHAT MOT BAN direct-iteration
     (hop dong "khong bao gio pull TAIL").
  B. tools.py  : loai ban sao chep cua _CANNED_FAILURE_RES +
     looks_like_canned_failure; neu detector bi mat hoan toan thi dung lai.
  C. server.py : sua import `looks_like_canned_failure,` bi lap thanh 1 dong.
  D. config.py : loai block `tool_stream/tool_delta_forward` bi lap.
  E. Kiem tra cuoi: py_compile + self-check don nhat + IN KET QUA
     `python -m unittest tests.test_tool_stream` va full suite.

Cach dung (Windows, tai thu muc goc repo -- canh thu muc gemini_web2api):
    python tool_flow_final_patch.py            # ap + tu chay test
    python tool_flow_final_patch.py --revert   # khoi phuc tu *.bakfinal
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = CONFIGF = TOOLS = None
for pkg in (ROOT / "gemini_web2api", ROOT / "gemini-web2api" / "gemini_web2api"):
    if (pkg / "server.py").exists():
        SERVER, CONFIGF, TOOLS = (pkg / "server.py", pkg / "config.py",
                                  pkg / "tools.py")
        break
if SERVER is None:
    print("ERROR: khong tim thay gemini_web2api/server.py (chay tu goc repo)")
    sys.exit(2)

BACKUP = ".bakfinal"

# ------------------------------------------------------------------- canonical code
GOOD_STREAMED_FN = '''def _streamed_generate(call_prompt, model_id, think_mode, file_refs,
                       extra, ticket, tools_active, attempt_no, on_delta=None):
    """generate_stream() collapsed to one string, with [timing] + early exit.

    Used by the tool-retry loops when CONFIG["tool_stream"] is on. Three jobs:
      * log first-delta timing so operator and client can tell "upstream slow"
        from "proxy hung";
      * stop reading upstream as soon as a *complete, parseable* tool_call
        fence has arrived (real parser primitives, not a backtick heuristic);
      * hand every delta to ``on_delta`` so the caller can keep the client's
        connection warm (heartbeat/ping) during a long buffered turn.

    CONTRACT (tests/test_tool_stream.py depends on it): the generator is
    iterated DIRECTLY in this thread and abandoned via close() at the
    early-exit point. A background pump thread would pre-pull the TAIL delta
    into a queue before the break lands and silently violate "never read the
    tail" -- that exact bug produced the failing assertion
    "'TAIL SHOULD NOT BE READ' unexpectedly found in seen".
    """
    chunks = []
    ttfb_done = False
    t0 = time.monotonic()
    gen = generate_stream(call_prompt, model_id, think_mode,
                          file_refs, extra, ticket)
    try:
        for delta in gen:
            if not ttfb_done and delta:
                ttfb_done = True
                log(f"[timing] tool attempt {attempt_no + 1}: first delta in "
                    f"{time.monotonic() - t0:.1f}s")
            chunks.append(delta)
            if on_delta is not None:
                try:
                    on_delta(delta)
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
    finally:
        close = getattr(gen, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
    return "".join(chunks)


'''

CANNED_TAIL = '''_CANNED_FAILURE_RES = (
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
                                                  for r in _CANNED_FAILURE_RES)


'''

CONFIG_BLOCK = '''    "tool_stream": False,
    # When tool_stream is on, also forward each streamed delta of a buffered
    # tool turn to the client (SSE ping / heartbeat) so agent clients see
    # per-part progress instead of 20-60 s of silence. Set false to keep the
    # old fully-buffered behaviour.
    "tool_delta_forward": True,
'''


# ----------------------------------------------------------------------- helpers
def top_level_spans(text, header):
    """[(start, end)] for every module-level block starting exactly with header."""
    spans = []
    for m in re.finditer(r"(?m)^" + re.escape(header), text):
        nxt = re.search(r"(?m)^(?:def |class |_?[A-Z][A-Za-z_]* = |\n\n\n)",
                        text[m.end():])
        end = m.end() + nxt.start() if nxt else len(text)
        spans.append((m.start(), end))
    return spans


def collapse_duplicates(path, header, label):
    """Remove all but the LAST module-level copy of `header`."""
    src = path.read_text(encoding="utf-8")
    spans = top_level_spans(src, header)
    if len(spans) <= 1:
        print(f"OK       {path.name}: {label} ({len(spans)} copy)")
        return False
    out, prev = [], 0
    for start, end in spans[:-1]:
        out.append(src[prev:start]); prev = end
    out.append(src[prev:])
    path.write_text("".join(out), encoding="utf-8")
    print(f"PATCHED  {path.name}: {label} -- removed {len(spans)-1} duplicate "
          "copy(ies), kept the final one")
    return True


def ensure_single_import(path, line, label):
    src = path.read_text(encoding="utf-8")
    n = src.count(line)
    if n == 1:
        print(f"OK       {path.name}: {label}")
        return False
    while src.count(line) > 1:
        i = src.find(line)
        src = src[:i] + src[i + len(line):]
    if src.count(line) == 0:  # never present -> add it
        anchor = "    looks_like_extension_refusal,\n"
        src = src.replace(anchor, anchor + line, 1)
    path.write_text(src, encoding="utf-8")
    print(f"PATCHED  {path.name}: {label} -- normalised to exactly one line")
    return True


def force_canonical_streamed(path):
    """Exactly ONE _streamed_generate, and it must be the direct-iteration one."""
    src = path.read_text(encoding="utf-8")
    spans = top_level_spans(src, "def _streamed_generate")
    if len(spans) == 1:
        body = src[spans[0][0]:spans[0][1]]
        pump = ("Thread(target=_pump" in body or "queue_mod.Queue" in body
                or "q.get(timeout" in body or "Queue()" in body)
        if not pump:
            print("OK       server.py: single direct-iteration _streamed_generate")
            return False
        src = src[:spans[0][0]] + GOOD_STREAMED_FN + src[spans[0][1]:]
        path.write_text(src, encoding="utf-8")
        print("PATCHED  server.py: replaced pump-thread _streamed_generate "
              "with direct-iteration version")
        return True
    # multiple copies: drop all but last, then check the survivor
    out, prev = [], 0
    for start, end in spans[:-1]:
        out.append(src[prev:start]); prev = end
    out.append(src[prev:])
    src = "".join(out)
    spans = top_level_spans(src, "def _streamed_generate")
    body = src[spans[0][0]:spans[0][1]]
    pump = ("Thread(target=_pump" in body or "queue_mod.Queue" in body
            or "q.get(timeout" in body or "Queue()" in body)
    if pump:
        src = src[:spans[0][0]] + GOOD_STREAMED_FN + src[spans[0][1]:]
    path.write_text(src, encoding="utf-8")
    print(f"PATCHED  server.py: _streamed_generate -- removed {len(spans)-1} "
          f"duplicate(s){', replaced pump-thread survivor' if pump else ''}")
    return True


def ensure_canned_detector(path):
    src = path.read_text(encoding="utf-8")
    changed = False
    spans = top_level_spans(src, "_CANNED_FAILURE_RES = (")
    if len(spans) > 1:
        out, prev = [], 0
        for start, end in spans[:-1]:
            out.append(src[prev:start]); prev = end
        out.append(src[prev:])
        src = "".join(out); changed = True
        print("PATCHED  tools.py: _CANNED_FAILURE_RES -- removed duplicates")
    dspan = top_level_spans(src, "def looks_like_canned_failure")
    if len(dspan) > 1:
        out, prev = [], 0
        for start, end in dspan[:-1]:
            out.append(src[prev:start]); prev = end
        out.append(src[prev:])
        src = "".join(out); changed = True
        print("PATCHED  tools.py: looks_like_canned_failure -- removed duplicates")
    if "def looks_like_canned_failure" not in src:
        anchor = "def looks_like_missed_tool_call"
        src = src.replace(anchor, CANNED_TAIL + anchor, 1)
        changed = True
        print("PATCHED  tools.py: rebuilt missing canned detector")
    if not changed:
        print("OK       tools.py: canned detector unique")
    path.write_text(src, encoding="utf-8")
    return changed


def dedupe_config_block(path):
    src = path.read_text(encoding="utf-8")
    n = src.count(CONFIG_BLOCK)
    if n <= 1:
        print(f"OK       config.py: tool_stream block ({n} copy)")
        return False
    i = src.find(CONFIG_BLOCK)
    src = src[:i] + src[i + len(CONFIG_BLOCK):]
    path.write_text(src, encoding="utf-8")
    print(f"PATCHED  config.py: removed 1 duplicated tool_stream block "
          f"(was {n})")
    return True


# --------------------------------------------------------------------------- main
def run_tests():
    print("\n--- tests/test_tool_stream.py ---")
    r1 = subprocess.run([sys.executable, "-m", "unittest",
                         "tests.test_tool_stream", "-v"], cwd=str(ROOT))
    print("--- full suite ---")
    r2 = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                         "tests"], cwd=str(ROOT))
    return r1.returncode == 0 and r2.returncode == 0


def apply(revert):
    files = [p for p in (SERVER, TOOLS, CONFIGF) if p.exists()]
    if revert:
        rc = 0
        for p in files:
            bak = p.with_suffix(p.suffix + BACKUP)
            if bak.exists():
                shutil.copy2(bak, p); print(f"REVERTED {p.name}")
            else:
                print(f"SKIP revert: no backup for {p.name}"); rc = 1
        return rc

    for p in files:
        bak = p.with_suffix(p.suffix + BACKUP)
        if not bak.exists():
            shutil.copy2(p, bak); print(f"BACKUP   {p.name}")

    collapse_duplicates(SERVER, "def _complete_call_end", "_complete_call_end")
    collapse_duplicates(SERVER, "class _Heartbeat", "_Heartbeat")
    force_canonical_streamed(SERVER)
    ensure_canned_detector(TOOLS)
    ensure_single_import(SERVER, "    looks_like_canned_failure,\n",
                         "import canned detector")
    dedupe_config_block(CONFIGF)

    for p in files:
        if subprocess.run([sys.executable, "-m", "py_compile", str(p)]).returncode:
            print(f"ERROR: py_compile failed for {p.name}"); return 1
    print("py_compile OK")

    src = SERVER.read_text(encoding="utf-8")
    ts = TOOLS.read_text(encoding="utf-8")
    cfg = CONFIGF.read_text(encoding="utf-8")
    checks = [("def _complete_call_end", src, 1),
              ("def _streamed_generate", src, 1),
              ("class _Heartbeat", src, 1),
              ("    looks_like_canned_failure,\n", src, 1),
              ("def looks_like_canned_failure", ts, 1),
              ("_CANNED_FAILURE_RES = (", ts, 1),
              ('"tool_stream"', cfg, 1)]
    ok = True
    for needle, hay, want in checks:
        got = hay.count(needle)
        if got != want:
            print(f"FAIL self-check: {needle!r} x{got} (want {want})"); ok = False
    if "Thread(target=_pump" in src or "Queue()" in src:
        print("FAIL self-check: pump-thread remnants still present"); ok = False
    print("self-check", "OK" if ok else "FAILED")
    if not ok:
        print("\nGhi chu: neu FAIL la do file goc cua ban lech kha xa (anchor "
              "khong thay), gui lai output nay de toi chinh xac.")
        return 1
    return 0 if run_tests() else 1


if __name__ == "__main__":
    sys.exit(apply("--revert" in sys.argv))
