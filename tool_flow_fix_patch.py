#!/usr/bin/env python3
"""tool_flow_fix_patch.py — sửa 1 hàm trong server.py sau khi tool_flow_patch chạy.

Nguyên nhân test fail (đã reproduce được):
    NEW_STREAMED_FN của tool_flow_patch dùng pump-thread + queue, luôn TÁCH
    generator ra khỏi thread request -> khi hàng đợi delta kế tiếp đã sẵn
    trong queue thì main loop vẫn append + check SAU KHI pull, nên delta
    "TAIL" bị pull trước lúc early-exit kịp set -> assertNotIn(...seen...) fail.
    Test đòi hỏi hợp đồng cũ: abandon generator ngay tại chỗ fence đóng.

Bản fix này thay _streamed_generate bằng bản iterate TRỰC TIẾP trong thread
request (giống latency_stream_patch) nhưng giữ nguyên các cải tiến thật:
    * parser thật _complete_call_end cho early-exit (không còn heuristic ```);
    * on_delta sink cho heartbeat;
    * gen.close() trong finally để abandoned generator không rò socket.

Cách dùng (repo gốc = state 6eb72b1, CHƯA patch gì):
    python latency_stream_patch.py
    python tool_flow_patch.py
    python tool_flow_fix_patch.py        # bản này
    python -m unittest discover -s tests # phải OK (307 tests)

Đã verify trên /workspace: unpatched -> 3 fails; cả 3 patch -> 307 tests OK.
"""
import shutil
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent / "gemini_web2api" / "server.py"

# Toàn bộ thân hàm hiện tại (queue-pump) -- từ def tới hết return "".join(chunks)
OLD_BODY = '''def _streamed_generate(call_prompt, model_id, think_mode, file_refs,
                       extra, ticket, tools_active, attempt_no, on_delta=None):
    """generate_stream() collapsed to one string, with [timing] + early exit.

    Used by the tool-retry loops when CONFIG["tool_stream"] is on. Three jobs:
      * log first-delta timing so operator and client can tell "upstream slow"
        from "proxy hung";
      * stop reading upstream as soon as a *complete, parseable* tool_call
        fence has arrived (real parser primitives, not a backtick heuristic);
      * hand every delta to ``on_delta`` so the caller can keep the client's
        connection warm (heartbeat/ping) during a long buffered turn.

    The wrapped generator is iterated directly in this thread and abandoned
    at the early-exit point via close(), preserving the pre-patch contract
    that TAIL deltas are never pulled -- a background pump would defeat it.
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
    return "".join(chunks)'''


def find_old_queue_pump(src: str):
    """Locate the queue-pump version of _streamed_generate (tool_flow_patch B1)."""
    start = src.find("def _streamed_generate(")
    if start < 0:
        return None
    end = src.find("\n\n\nclass _Heartbeat:", start)
    if end < 0:
        return None
    body = src[start:end]
    if "_pump" not in body or "q.get(timeout" not in body:
        return None          # already direct-iteration -> nothing to do
    return start, end


def apply(revert: bool) -> int:
    if not SERVER.exists():
        print(f"ERROR: not found: {SERVER} (chạy từ gốc repo)")
        return 2
    bak = SERVER.with_suffix(".py.bak-fix")
    if revert:
        if bak.exists():
            shutil.copy2(bak, SERVER)
            print("REVERTED server.py từ .bak-fix")
        else:
            print("SKIP revert: không có .bak-fix")
        return 0

    src = SERVER.read_text(encoding="utf-8")

    if OLD_BODY in src:
        print("ALREADY  _streamed_generate là bản direct-iteration (đã fix)")
        return 0

    loc = find_old_queue_pump(src)
    if loc is None:
        print("SKIP     không thấy queue-pump _streamed_generate -- chắc bạn "
              "chưa chạy tool_flow_patch.py (hoặc version khác). Thứ tự đúng:\n"
              "         latency_stream_patch.py -> tool_flow_patch.py -> bản này")
        return 1

    start, end = loc
    if not bak.exists():
        shutil.copy2(SERVER, bak)
        print(f"BACKUP   server.py -> {bak.name}")
    src = src[:start] + OLD_BODY + src[end:]
    SERVER.write_text(src, encoding="utf-8")
    print("PATCHED  _streamed_generate -> direct iteration + real-parser "
          "early-exit + gen.close()")

    rc = subprocess.run([sys.executable, "-m", "py_compile", str(SERVER)])
    if rc.returncode != 0:
        print("ERROR: py_compile fail!")
        return 1
    print("py_compile OK")
    print("Giờ chạy:  python -m unittest discover -s tests   (phải OK)")
    return 0


if __name__ == "__main__":
    sys.exit(apply("--revert" in sys.argv))
