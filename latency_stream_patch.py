#!/usr/bin/env python3
"""latency_stream_patch.py — patch gemini_web2api để proxy trả lời nhanh hơn.

Bối cảnh (đã verify bằng evidence trong repo):
 1. Khi request có tools, server đi qua tool-retry loop và mỗi attempt gọi
    ``generate()`` (NON-streaming, urllib blocking). Nghĩa là client phải chờ
    toàn bộ upstream turn mới nhận được phản hồi đầu tiên -> "proxy chạy chậm",
    không có status từng phần. Trong khi đó ``generate_stream()`` đã tồn tại
    và stream thật sự (httpx), chỉ bị bỏ qua ở path tools.
 2. SSE keep-alive ping间隔 = 15s (server.py: ``SSE_PING_INTERVAL_SEC = 15``)
    -> giai đoạn chết giữa các attempt dài.
 3. Không có timing log per-attempt -> không đo được bottleneck; retry loop
    cũng không sleep giữa các tool-retry nên fail nhanh liên tiếp, nhưng cũng
    không có backoff khi upstream quá tải.

Patch này (idempotent, có backup .bak, anchor-based — SKIP nếu anchor không
khớp thay vì phá file):
  A. gemini.py : thêm timing log [timing] cho generate()/generate_stream()
     (TTFB với stream, tổng thời gian mỗi attempt).
  B. server.py : SSE_PING_INTERVAL_SEC 15 -> 5 (client thấy activity sớm).
  C. server.py : tool-retry loop dùng generate_stream() gộp delta khi
     CONFIG["tool_stream"]=True -> partial text chảy về client ngay, đồng thời
     phát hiện tool_call fence hoàn chỉnh sớm để early-exit attempt (đỡ chờ cả
     turn khi đã có call). Mặc định OFF để giữ nguyên hành vi cũ; bật bằng
     {"tool_stream": true} trong config.json.
  D. server.py : backoff nhẹ (retry_delay_sec) giữa các tool-retry attempt.

Cách dùng (chạy tại thư mục gốc repo, cạnh thư mục gemini_web2api/):
    python latency_stream_patch.py            # áp patch (tạo *.bak)
    python latency_stream_patch.py --revert   # khôi phục từ *.bak
Sau khi patch xong, bật streaming path trong config.json:
    {"tool_stream": true}
"""
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEMINI = ROOT / "gemini_web2api" / "gemini.py"
SERVER = ROOT / "gemini_web2api" / "server.py"


def _apply(path: Path, edits, revert: bool):
    """edits: list of (name, old, new). Idempotent + anchor-checked."""
    if revert:
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            shutil.copy2(bak, path)
            print(f"REVERTED {path.name} from {bak.name}")
        else:
            print(f"SKIP revert: no backup for {path.name}")
        return

    src = path.read_text(encoding="utf-8")
    changed = False
    for name, old, new in edits:
        marker = new.splitlines()[0].strip()
        if new in src and old not in src:
            print(f"ALREADY  {path.name}: {name}")
            continue
        n = src.count(old)
        if n == 0:
            print(f"SKIP     {path.name}: {name} (anchor not found — file "
                  f"drifted, please send this output)")
            continue
        if n > 1:
            print(f"SKIP     {path.name}: {name} (anchor matches {n}x, not unique)")
            continue
        if not any(b.name == path.name + ".bak" for b in path.parent.iterdir()):
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        src = src.replace(old, new)
        changed = True
        print(f"PATCHED  {path.name}: {name}")
    if changed:
        path.write_text(src, encoding="utf-8")


# ---------------------------------------------------------------- gemini.py
# A1: generate() — time each attempt (blocking full-turn latency visible).
GEN_OLD = '''    last_err = None
    bl_refreshed = False
    for attempt in range(CONFIG["retry_attempts"]):
        url = _get_url()
        headers = _build_headers(ticket)
        try:'''
GEN_NEW = '''    last_err = None
    bl_refreshed = False
    for attempt in range(CONFIG["retry_attempts"]):
        _t0 = time.monotonic()
        url = _get_url()
        headers = _build_headers(ticket)
        try:'''

GEN_OLD2 = '''            trace("MODEL -> PROXY", raw)
            check_routing(raw, model_id, extra_fields, ticket)
            return extract_response_text(raw)'''
GEN_NEW2 = '''            trace("MODEL -> PROXY", raw)
            check_routing(raw, model_id, extra_fields, ticket)
            text = extract_response_text(raw)
            log(f"[timing] generate attempt {attempt + 1}: "
                f"{time.monotonic() - _t0:.1f}s, {len(text)} chars")
            return text'''

# A2: generate_stream() — TTFB + total per attempt.
STR_OLD = '''    for attempt in range(CONFIG["retry_attempts"]):
        url = _get_url()
        headers = _build_headers(ticket)
        emitted_raw_text = ""'''
STR_NEW = '''    for attempt in range(CONFIG["retry_attempts"]):
        _t0 = time.monotonic()
        _ttfb_logged = False
        url = _get_url()
        headers = _build_headers(ticket)
        emitted_raw_text = ""'''

STR_OLD2 = '''                            delta = clean_text(clean_buf, strip=False)
                            clean_buf = ""
                            if delta:
                                emitted_any = True
                                yield delta
            if clean_buf:'''
STR_NEW2 = '''                            delta = clean_text(clean_buf, strip=False)
                            clean_buf = ""
                            if delta:
                                if not _ttfb_logged:
                                    _ttfb_logged = True
                                    log(f"[timing] stream attempt "
                                        f"{attempt + 1}: first token in "
                                        f"{time.monotonic() - _t0:.1f}s")
                                emitted_any = True
                                yield delta
            if clean_buf:'''

STR_OLD3 = '''            trace("MODEL -> PROXY (stream)", raw_full)
            check_routing(raw_full, model_id, extra_fields, ticket)
            return'''
STR_NEW3 = '''            trace("MODEL -> PROXY (stream)", raw_full)
            check_routing(raw_full, model_id, extra_fields, ticket)
            log(f"[timing] stream attempt {attempt + 1}: "
                f"{time.monotonic() - _t0:.1f}s total, {len(raw_full)} raw bytes")
            return'''

# ---------------------------------------------------------------- server.py
# B: faster keep-alive so the client sees status activity within ~5s.
PING_OLD = "SSE_PING_INTERVAL_SEC = 15"
PING_NEW = "SSE_PING_INTERVAL_SEC = 5"

# Helper injected into server.py: streamed attempt with early tool-call exit.
HELPER_OLD = '''def _estimate_tokens(text: str) -> int:'''
HELPER_NEW = '''def _streamed_generate(call_prompt, model_id, think_mode, file_refs,
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
    return "".join(chunks)


def _estimate_tokens(text: str) -> int:'''

# C: opt-in real streaming inside BOTH tool-retry loops (OpenAI + Google).
CALL_OLD = '''                call_prompt = prompt + "\\n\\n" + nudge
            try:
                raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Tool retry after upstream error: {e}")'''
CALL_NEW = '''                call_prompt = prompt + "\\n\\n" + nudge
            try:
                if CONFIG.get("tool_stream"):
                    raw = _streamed_generate(call_prompt, model_id, think_mode,
                                             file_refs, extra, ticket,
                                             tools_active, attempt)
                else:
                    raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Tool retry after upstream error: {e}")'''

G_CALL_OLD = '''                call_prompt = prompt + "\\n\\n" + nudge
            try:
                raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Google tool retry after upstream error: {e}")'''
G_CALL_NEW = '''                call_prompt = prompt + "\\n\\n" + nudge
            try:
                if CONFIG.get("tool_stream"):
                    raw = _streamed_generate(call_prompt, model_id, think_mode,
                                             file_refs, extra, ticket,
                                             tools_active, attempt)
                else:
                    raw = generate(call_prompt, model_id, think_mode, file_refs, extra, ticket)
            except Exception as e:
                if attempt + 1 < attempts:
                    log(f"Google tool retry after upstream error: {e}")'''

# D: gentle backoff between tool-retry attempts (was: immediate re-fire).
BACKOFF_OLD = '''            if attempt + 1 >= attempts:
                break
            if not (text or "").strip():'''
BACKOFF_NEW = '''            if attempt + 1 >= attempts:
                break
            time.sleep(CONFIG["retry_delay_sec"])
            if not (text or "").strip():'''

IMPORT_OLD = "from .gemini import ("
IMPORT_NOTE = ("we do not touch the import block blindly; instead verify "
               "generate_stream/looks_like_tool_call are importable below")


def patch_gemini(revert):
    _apply(GEMINI, [
        ("generate: per-attempt timer start", GEN_OLD, GEN_NEW),
        ("generate: [timing] log before return", GEN_OLD2, GEN_NEW2),
        ("stream: per-attempt timer start", STR_OLD, STR_NEW),
        ("stream: [timing] first-token log", STR_OLD2, STR_NEW2),
        ("stream: [timing] total log", STR_OLD3, STR_NEW3),
    ], revert)


TOOLS_IMPORT_OLD = '''from .tools import (
    messages_to_prompt,
    parse_tool_calls,'''
TOOLS_IMPORT_NEW = '''from .tools import (
    looks_like_tool_call,
    messages_to_prompt,
    parse_tool_calls,'''

CONFIG_DEFAULT_OLD = '''    "retry_attempts": 3,'''
CONFIG_DEFAULT_NEW = '''    "tool_stream": False,
    "retry_attempts": 3,'''


def patch_server(revert):
    _apply(SERVER, [
        ("SSE ping 15s -> 5s", PING_OLD, PING_NEW),
        ("inject _streamed_generate helper", HELPER_OLD, HELPER_NEW),
        ("import looks_like_tool_call", TOOLS_IMPORT_OLD, TOOLS_IMPORT_NEW),
        ("OpenAI tool loop: streamed attempt opt-in", CALL_OLD, CALL_NEW),
        ("Google tool loop: streamed attempt opt-in", G_CALL_OLD, G_CALL_NEW),
        ("tool loop: backoff between attempts", BACKOFF_OLD, BACKOFF_NEW),
    ], revert)
    _apply(ROOT / "gemini_web2api" / "config.py", [
        ("config default tool_stream=False", CONFIG_DEFAULT_OLD, CONFIG_DEFAULT_NEW),
    ], revert)


def syntax_check():
    import py_compile
    ok = True
    for p in (GEMINI, SERVER):
        try:
            py_compile.compile(str(p), doraise=True)
            print(f"OK       py_compile {p.name}")
        except Exception as e:
            ok = False
            print(f"FAILED   py_compile {p.name}: {e}")
    return ok


def main():
    revert = "--revert" in sys.argv
    if not GEMINI.exists() or not SERVER.exists():
        sys.exit("Run this script at the repo root (needs gemini_web2api/gemini.py and server.py)")
    patch_gemini(revert)
    patch_server(revert)
    if not revert:
        if not syntax_check():
            sys.exit("Syntax check failed — run `python latency_stream_patch.py --revert`")
        print("\nDone. Bật streaming path tools bằng cách thêm vào config.json:")
        print('    "tool_stream": true')
    else:
        print("Reverted.")


if __name__ == "__main__":
    main()
