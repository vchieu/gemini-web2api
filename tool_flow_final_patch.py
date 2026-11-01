#!/usr/bin/env python3
"""tool_flow_final_patch.py -- patch "cuoi cung" cho loi test_stream_attempt_exits_early_on_complete_fence.

Tinh trang /workspace (da pull/clone, commit cua ban):
    - Full suite:  python -m unittest discover -s tests  ->  Ran 307 tests OK
    - Rieng:       python -m unittest tests.test_tool_stream -v -> 4 tests OK
    Nghia la NGUON DUNG; loi con lai tren may ban 100% do PATCH AP DUP
    (server.py co 2 lan _complete_call_end/_streamed_generate/_Heartbeat,
     tools.py co 2 lan looks_like_canned_failure, config.py co 2 lan
     tool_stream). Ban patch cu khong idempotent hoan toan: chay 2 lan thi
     dinh nghia bi trung lap va ban nao nam sau moi du dung. Neu mot trong
     cac ban duoc ap la ban pump-thread (queue + worker thread), delta TAIL
     bi pull san vao queue TRUOC khi early-exit kip --> chinh xac la loi ban
     thay ('TAIL SHOULD NOT BE READ' xuat hien trong `seen`).

Patch nay (idempotent, anchor-based, co dedupe, co --revert):
  A. server.py : loai ban SAO CHEP dau tien cua _complete_call_end /
     _streamed_generate / _Heartbeat (giu ban cuoi = ban direct-iteration
     dung hop dong "khong bao gio pull TAIL").
  B. tools.py  : loai ban sao chep cua _CANNED_FAILURE_RES +
     looks_like_canned_failure.
  C. server.py : loai dong import `looks_like_canned_failure,` bi lap.
  D. config.py : loai block `tool_stream/tool_delta_forward` bi lap.
  E. Hanh vi bat buoc (kiem tra bang test, xem verify() ben duoi):
     _streamed_generate KHONG duoc dung pump-thread/queue; phai iterate
     generator truc tiep va break ngay khi fence dong -> TAIL khong bi pull.

Cach dung (Windows, tai thu muc goc repo):
    python tool_flow_final_patch.py            # ap + kiem tra + test
    python tool_flow_final_patch.py --revert   # khoi phuc tu *.bak
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG_CANDIDATES = [ROOT / "gemini_web2api", ROOT / "gemini-web2api" / "gemini_web2api"]
SERVER = CONFIGF = TOOLS = None
for pkg in PKG_CANDIDATES:
    if (pkg / "server.py").exists():
        SERVER, CONFIGF, TOOLS = (pkg / "server.py", pkg / "config.py",
                                  pkg / "tools.py")
        break
if SERVER is None:
    print("ERROR: khong tim thay gemini_web2api/server.py (chay tu goc repo)")
    sys.exit(2)

BACKUP = ".bakfinal"


# --------------------------------------------------------------------------- helpers
def dup_blocks(text, header):
    """Return [(start, end)] of every module-level block starting with `header`
    and running until the next top-level def/class or a blank line followed by
    a top-level statement (same rules for functions and classes)."""
    spans = []
    for m in re.finditer(r"(?m)^" + re.escape(header), text):
        start = m.start()
        nxt = re.search(r"(?m)^(?:def |class )", text[m.end():])
        end = m.end() + nxt.start() if nxt else len(text)
        spans.append((start, end))
    return spans


def strip_first_dup(path, header, label):
    """If `header` appears >1 times at module level, delete all but the LAST."""
    src = path.read_text(encoding="utf-8")
    spans = dup_blocks(src, header)
    if len(spans) <= 1:
        print(f"OK       {path.name}: {label} ({len(spans)} copy)")
        return False
    # keep the last definition (Python binds it last anyway); drop earlier ones
    out, prev = [], 0
    for start, end in spans[:-1]:
        out.append(src[prev:start]); prev = end
    out.append(src[prev:])
    path.write_text("".join(out), encoding="utf-8")
    print(f"PATCHED  {path.name}: {label} -- removed {len(spans)-1} duplicate "
          f"copy(ies), kept the final one")
    return True


def strip_dup_line(path, line, label):
    src = path.read_text(encoding="utf-8")
    n = src.count(line)
    if n <= 1:
        print(f"OK       {path.name}: {label} ({n} copy)")
        return False
    first = src.find(line)
    src = src[:first] + src[first + len(line):]
    path.write_text(src, encoding="utf-8")
    print(f"PATCHED  {path.name}: {label} -- removed 1 duplicate line")
    return True


CONFIG_BLOCK_DUP = '''    "tool_stream": False,
    # When tool_stream is on, also forward each streamed delta of a buffered
    # tool turn to the client (SSE ping / heartbeat) so agent clients see
    # per-part progress instead of 20-60 s of silence. Set false to keep the
    # old fully-buffered behaviour.
    "tool_delta_forward": True,
'''


def apply(revert):
    files = [p for p in (SERVER, TOOLS, CONFIGF) if p.exists()]
    if revert:
        rc = 0
        for p in files:
            bak = p.with_suffix(p.suffix + BACKUP)
            if bak.exists():
                shutil.copy2(bak, p)
                print(f"REVERTED {p.name}")
            else:
                print(f"SKIP revert: no backup for {p.name}")
                rc = 1
        return rc

    modified = False
    for p in files:
        bak = p.with_suffix(p.suffix + BACKUP)
        if not bak.exists():
            shutil.copy2(p, bak)
            print(f"BACKUP   {p.name}")

    # A/B/C/D -- dedupe everything the previous patches could have applied twice
    for header, label in (("def _complete_call_end", "_complete_call_end"),
                          ("def _streamed_generate", "_streamed_generate"),
                          ("class _Heartbeat", "_Heartbeat")):
        modified |= strip_first_dup(SERVER, header, label)
    modified |= strip_first_dup(TOOLS, "def looks_like_canned_failure",
                                "looks_like_canned_failure")
    modified |= strip_first_dup(TOOLS, "_CANNED_FAILURE_RES = (",
                                "_CANNED_FAILURE_RES")
    modified |= strip_dup_line(SERVER, "    looks_like_canned_failure,\n",
                               "import canned detector")

    src = CONFIGF.read_text(encoding="utf-8")
    if src.count(CONFIG_BLOCK_DUP) > 1:
        i = src.find(CONFIG_BLOCK_DUP)
        CONFIGF.write_text(src[:i] + src[i + len(CONFIG_BLOCK_DUP):],
                           encoding="utf-8")
        print("PATCHED  config.py: removed duplicated tool_stream block")
        modified = True
    else:
        print(f"OK       config.py: tool_stream block unique "
              f"({src.count(CONFIG_BLOCK_DUP)})")

    # E -- guard: the surviving _streamed_generate must be the direct-iteration
    # version. If a pump-thread build survived, replace it wholesale.
    src = SERVER.read_text(encoding="utf-8")
    bad = ("Thread(target=_pump" in src or "q.get(timeout" in src
           or "queue_mod.Queue" in src)
    if bad:
        start = src.index("def _streamed_generate")
        nxt = re.search(r"(?m)^(?:def |class )", src[start + 5:])
        end = start + 5 + nxt.start() if nxt else len(src)
        replacement = '''def _streamed_generate(call_prompt, model_id, think_mode, file_refs,
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
        SERVER.write_text(src[:start] + replacement + src[end:], encoding="utf-8")
        print("PATCHED  server.py: replaced pump-thread _streamed_generate "
              "with direct-iteration version")
        modified = True
    else:
        print("OK       server.py: _streamed_generate is direct-iteration "
              "(no pump thread)")

    for p in files:
        r = subprocess.run([sys.executable, "-m", "py_compile", str(p)])
        if r.returncode:
            print(f"ERROR: py_compile failed for {p.name}"); return 1
    print("py_compile OK")

    # Self-checks (no mocks needed): counts must be exactly 1 now
    src = SERVER.read_text(encoding="utf-8")
    ts = TOOLS.read_text(encoding="utf-8")
    checks = [("def _complete_call_end", src, 1), ("def _streamed_generate", src, 1),
              ("class _Heartbeat", src, 1), ("looks_like_canned_failure,\n", src, 1),
              ("def looks_like_canned_failure", ts, 1), ("_CANNED_FAILURE_RES = (", ts, 1)]
    ok = True
    for needle, hay, want in checks:
        got = hay.count(needle)
        if got != want:
            print(f"FAIL self-check: {needle!r} x{got} (want {want})"); ok = False
    cfg = CONFIGF.read_text(encoding="utf-8")
    if cfg.count('"tool_stream"') != 1:
        print("FAIL self-check: config.py tool_stream not unique"); ok = False
    print("self-check", "OK" if ok else "FAILED")

    print("\n--- chay test ---")
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                       cwd=str(ROOT))
    if r.returncode == 0 and ok and modified is not None:
        print('\nDone. Nho: "tool_stream": true trong config.json de bat stream.')
    return r.returncode or (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(apply("--revert" in sys.argv))
