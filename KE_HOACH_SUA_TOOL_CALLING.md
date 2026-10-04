# Kế hoạch sửa tool calling cho gemini-web2api

> Kế hoạch dựa trên việc đọc source, **chưa chạy thực tế**. Giai đoạn 0 là bắt buộc: nó xác nhận giả thuyết trước khi sửa.

## Bối cảnh

Dùng proxy local (gemini-web2api) để chạy opencode. Triệu chứng: có lúc model gọi tool, có lúc không (ví dụ "đọc README" khi thì có tool call, khi thì model trả lời như thể đã đọc mà không gọi tool nào). Hành vi "lúc được lúc không" khớp với các lỗi chỉ kích hoạt theo nội dung từng request.

## Nguyên nhân nghi vấn (xếp theo khả năng)

1. **`extract_response_text` (gemini.py) chọn text dài nhất.** Nếu upstream trả thêm thinking summary/draft trong cùng response, khối `tool_call` ngắn sẽ thua và bị vứt.
2. **Regex parser (tools.py) dừng ở dấu ``` đầu tiên.** Tool `write`/`edit` có arguments chứa markdown/code thì JSON bị cắt, `json.loads` fail, block bị drop.
3. **Tool definitions bị cắt parameters** khi JSON > `PROMPT_MAX_BYTES // 3`. Tool của opencode có description rất dài nên dễ vượt ngưỡng, model không còn biết tên tham số. Ngoài ra `_join_prompt_parts` có thể loại hẳn khối `# Tool Use` khi lịch sử/system prompt quá lớn.
4. **Prompt tool yếu và đặt sai chỗ:** chỉ ở đầu prompt, không cấm bịa kết quả đọc file, không nói tool đã kết nối thật. Nhánh Google dùng `build_tool_prompt` đã tinh chỉnh, nhánh OpenAI thì chưa.
5. **Parser chỉ nhận fence `tool_call`**, trong khi Gemini hay viết ```` ```json ````, ```` ```function_call ```` hoặc JSON trần.
6. **Không retry khi model "nói mà không làm"** (chỉ retry khi `tool_choice=required`). Khi có tools, server chờ `generate()` xong mới gửi header SSE nên request thinking lâu có thể làm client timeout.

## Nguyên tắc chung (đưa cho model nhỏ)

- Làm **tuần tự** từng giai đoạn. Mỗi giai đoạn chạy `python -m unittest` và phải xanh rồi mới sang giai đoạn sau.
- Không refactor ngoài phạm vi, không đổi API công khai, không đổi tên hàm đang được import.
- Mỗi thay đổi hành vi có công tắc trong `config.py` (`DEFAULT_CONFIG`) để rollback nhanh.
- Mỗi giai đoạn thêm test vào `tests/test_modular_sync.py` theo style hiện có (mock `generate`, helper `_wrb_line`, `_wrb_line_parts`).
- Mỗi giai đoạn một commit riêng.

---

## Giai đoạn 0: Đo trước khi sửa

**Mục tiêu:** biết lỗi nằm ở model, parser, extract hay prompt.

1. Đặt `"log_file": "gemini.log"` trong `config.json`, chạy opencode với 3 kịch bản:
   - "đọc README" (đơn giản)
   - "review source" (nhiều tool call)
   - yêu cầu ghi file có markdown (`write`/`edit`)
2. Với mỗi request lỗi, đối chiếu dòng log `upstream: N chars, X tool_call marker(s)`:

| Log | Nghĩa | Giai đoạn sửa |
|---|---|---|
| `0 marker`, text là lời từ chối hoặc "để mình đọc..." | Lỗi prompt hoặc thiếu retry | 3, 4 |
| `marker(s) present but no tool call was parsed` | Lỗi parser | 1 |
| Text ngắn bất thường | Lỗi chọn text | 2 |
| `Prompt truncated` + model không biết tool nào | Tool definitions bị cắt | 3 |

3. Thêm log debug tạm trong `generate()` (gemini.py): khi `CONFIG.get("debug_raw")` bật, ghi `raw[:20000]` vào file riêng. Mục đích: xem một response thật có mấy phần tử trong `inner[4]` và phần tử nào là thinking.

**Tiêu chí xong:** có ít nhất 3 mẫu raw/log thật, kết luận được lỗi nào chiếm đa số. Lưu mẫu (đã xoá cookie) vào `tests/fixtures/` làm test hồi quy.

---

## Giai đoạn 1: Parser chịu được ``` bên trong arguments

**File:** `tools.py`, hàm `parse_tool_calls`.

**Việc làm:**

1. Thêm `_iter_tool_blocks(text)` dùng `json.JSONDecoder().raw_decode`, yield `(start, end, data)`:

```python
_FENCE_OPEN = re.compile(r'```(?:tool_call|function_call|tool_code|json)[ \t]*\n?')
_FENCE_CLOSE = re.compile(r'\s*```')

def _iter_tool_blocks(text):
    dec, pos = json.JSONDecoder(), 0
    while True:
        m = _FENCE_OPEN.search(text, pos)
        if not m:
            return
        j = m.end()
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        try:
            data, end = dec.raw_decode(text, j)
        except ValueError:
            pos = m.end()
            continue
        c = _FENCE_CLOSE.match(text, end)
        end = c.end() if c else end
        yield m.start(), end, data
        pos = end
```

2. Fence `tool_call`, `function_call`, `tool_code`: nhận như bình thường.
3. Fence `json`: chỉ nhận khi `allowed_names` không rỗng, `data["name"] in allowed_names`, và payload có khoá arguments (hoặc khoá khớp `tool_schemas`). Điều kiện này tránh nuốt nhầm ví dụ JSON bình thường trong câu trả lời.
4. `parse_tool_calls` lặp qua `_iter_tool_blocks` thay cho `re.finditer`. Giữ nguyên logic bên dưới: suy luận name, `extract_arguments`, `_arguments_json`, giữ lại block lỗi trong clean text.
5. Áp dụng tương tự cho `parse_google_function_calls`.

**Test mới:**
- arguments chứa ``` trong chuỗi vẫn parse đúng.
- fence `json` với tên đã khai báo thì được nhận.
- fence `json` với tên không khai báo thì giữ nguyên trong text.
- JSON không fence đứng một mình thì không bị nhận.
- Toàn bộ test cũ về `parse_tool_calls` vẫn xanh.

---

## Giai đoạn 2: Không để thinking thắng tool_call

**File:** `gemini.py`, hàm `extract_response_text`.

> Chỉ làm **sau** khi Giai đoạn 0 xác nhận cấu trúc raw.

**Việc làm:**

1. Chỉ xét `texts[0]` của mỗi dòng (giống cách `generate_stream` coi `index > 0` là block riêng), chọn bản dài nhất vì chúng cộng dồn:

```python
best = ""
for line in raw.split("\n"):
    texts = _extract_texts_from_line(line)
    if texts and len(texts[0]) > len(best):
        best = texts[0]
return clean_text(best)
```

2. Nếu mẫu thật cho thấy thinking nằm ở `texts[0]`, đổi chiến lược: loại phần tử đã khớp tiền tố với phần tử khác, ưu tiên phần tử chứa fence tool.
3. Thêm `CONFIG["extract_primary_only"]` (mặc định True) để rollback.

**Test mới:**
- `_wrb_line_parts(["thinking dài " * 20, '```tool_call\n{"name":"a","arguments":{}}\n```'])` cho kết quả chứa tool_call.
- Văn bản cộng dồn nhiều dòng vẫn ra bản cuối.
- Test hiện có của `generate_stream` không đổi.

---

## Giai đoạn 3: Tool definitions không bao giờ bị mất, prompt đủ mạnh

**File:** `tools.py` (`messages_to_prompt`, `_join_prompt_parts`, `build_tool_prompt`).

### 3a. Ghim khối tool khi cắt prompt

`_join_prompt_parts` giữ tin nhắn mới nhất trước rồi mới dùng ngân sách còn lại cho đầu prompt. System prompt của opencode rất lớn nên khối `# Tool Use` có thể bị loại hoàn toàn.

- Thêm tham số `pinned_head: str`. Trừ ngân sách của khối này **trước**, rồi mới phân bổ cho tail.
- Trong `messages_to_prompt`, tách khối tool khỏi `parts` và truyền vào làm `pinned_head`.
- Giới hạn khối ghim tối đa khoảng 40% `PROMPT_MAX_BYTES`.

### 3b. Nén tool thay vì bỏ parameters

```python
def _compact_tool(t, desc_max=300):
    p = t.get("parameters") or {}
    props, req = p.get("properties", {}), set(p.get("required", []))
    args = ", ".join(f'{k}{"" if k in req else "?"}: {v.get("type","any")}'
                     for k, v in props.items())
    return f'- {t["name"]}({args}): {(t.get("description") or "")[:desc_max]}'
```

- Quy trình: thử JSON đầy đủ; nếu quá ngưỡng dùng dạng compact; nếu vẫn quá thì cắt description xuống 150 rồi 80 ký tự.
- **Không bao giờ** bỏ tên, kiểu và required của parameters.
- Sửa log để nói rõ "đã nén" thay vì "dropped".

### 3c. Viết lại prompt `# Tool Use`

Nội dung bắt buộc:

- (a) Tool đã kết nối thật với máy người dùng và sẽ chạy ngay khi được gọi.
- (b) Không được nói là không có quyền truy cập tool hay môi trường bị hạn chế.
- (c) Không khẳng định đã đọc/chạy gì nếu chưa có `[Tool result ...]`. Muốn biết nội dung file hay kết quả lệnh thì **phải** gọi tool.
- (d) Khi gọi tool, chỉ xuất block `tool_call`, mọi tham số nằm trong `"arguments"`.
- (e) Có thể gọi nhiều block trong một lượt.

Thêm một dòng nhắc ngắn ở **cuối prompt** (sau tin nhắn cuối) khi có tools và `tool_choice != "none"`. Dùng chung nội dung cho nhánh Google (gộp vào `build_tool_prompt`).

**Test mới:**
- 400 tool giả: prompt vẫn chứa tên tham số và `<= PROMPT_MAX_BYTES`.
- Lịch sử khổng lồ: `# Tool Use` và câu hỏi cuối đều còn trong prompt.
- Dòng nhắc cuối có mặt khi có tools, vắng mặt khi không có tools hoặc `tool_choice="none"`.
- Test cắt prompt hiện có vẫn xanh.

---

## Giai đoạn 4: Retry khi model "nói mà không làm"

**File:** `server.py` (`_handle_chat`), helper trong `tools.py`.

1. Thêm `looks_like_missed_tool_call(text) -> bool`, True khi **tất cả** thoả:
   - text khớp mẫu từ chối/ý định: "không thể truy cập", "cannot access", "don't have access", "không có quyền", "để mình đọc", "let me read/check/look", hoặc kết thúc bằng `:`
   - text ngắn (ví dụ < 600 ký tự)
   - không chứa fence tool
2. Trong `_handle_chat`, nếu có tools, `tool_choice != "none"`, không có `tool_calls` và hàm trên trả True thì retry **tối đa 1 lần** với `"\n\nIMPORTANT: Respond with a tool_call block ONLY."`.
3. Công tắc `CONFIG["tool_retry_on_miss"]` (mặc định True). Log rõ mỗi lần retry để theo dõi tỉ lệ.
4. Chỉ retry khi `generate` thành công. Gộp với retry của `required_tool` vào một vòng lặp, `attempts` tối đa 2.

**Test mới:**
- Lần 1 trả "Để mình đọc file:", lần 2 trả tool_call: kết quả có `tool_calls`, `generate` được gọi đúng 2 lần.
- Câu trả lời cuối bình thường, dài, không khớp mẫu: không retry.
- `tool_retry_on_miss=False`: không retry.

---

## Giai đoạn 5: Keepalive SSE cho request có tools

**File:** `server.py`.

**Vấn đề:** khi `stream=True` kèm tools, server chờ `generate()` xong mới gửi header. Với model thinking, client có thể timeout vì im lặng.

**Việc làm:**

1. Với nhánh stream có tools: gọi `_start_sse()` và gửi chunk `role` **ngay**, trước khi `generate`.
2. Chạy `generate` trong `threading.Thread`, kết quả vào biến chia sẻ. Vòng chính `done.wait(10)` và gửi `": keepalive\n\n"` mỗi lần chưa xong.
3. Vì header 200 đã gửi, lỗi upstream phải báo bằng `_sse_event_error` rồi `_sse_done`, không dùng `send_api_error`.
4. Bắt `BrokenPipeError` và `ConnectionResetError` khi ghi keepalive. Client ngắt thì dừng chờ.
5. Retry của Giai đoạn 4 chạy bên trong thread, keepalive tiếp tục trong lúc đó.

**Test mới:**
- Mock `generate` có `time.sleep` ngắn, hạ khoảng keepalive xuống rất nhỏ qua config. Body phải chứa `: keepalive`, chunk role đứng đầu, kết thúc `[DONE]`, `tool_calls` đúng chỉ số.
- Upstream lỗi giữa chừng: có error event kèm `[DONE]`.

---

## Giai đoạn 6: Đồng bộ `/v1/responses`

**File:** `server.py` (`_handle_responses`).

1. Truyền `tool_schemas = tool_parameters(tools)` vào `parse_tool_calls` (hiện chưa truyền nên không suy luận được tên tool thiếu).
2. Gọi `_log_tool_trace`.
3. Áp dụng retry của Giai đoạn 4 (đưa logic dùng chung vào một hàm để chat và responses cùng gọi).
4. Xử lý `tool_choice` dạng flattened `{"type":"function","name":"x"}` cho `required_tool`.

**Test mới:** tool call thiếu `name` được suy luận trong nhánh responses; retry hoạt động ở responses.

---

## Giai đoạn 7: Nghiệm thu với opencode thật

Chạy lại ba kịch bản của Giai đoạn 0, mỗi kịch bản **5 lần** (lỗi ngẫu nhiên theo nội dung).

| Kịch bản | Đạt khi |
|---|---|
| "Đọc README" | Cả 5 lần có tool call `read`, không có câu "đã đọc" bịa |
| "Review source" | Gọi liên tiếp nhiều tool, mỗi lượt có tool result, kết thúc bằng tóm tắt |
| Ghi file có markdown | arguments không bị cắt, file ghi đúng |
| Model thinking, request dài | Không timeout phía client |

Tìm trong log: "Dropping tool_call block", "tool_call marker(s) present but no tool call was parsed", "Prompt truncated" và số lần retry. Retry nhiều nghĩa là prompt (Giai đoạn 3) vẫn chưa đủ mạnh.

---

## Thứ tự ưu tiên nếu ít thời gian

**0 → 1 → 3a → 3c → 4 → 2**, rồi mới đến 3b, 5, 6.

Lý do: 3a (tool bị cắt khỏi prompt) và 3c (prompt yếu) là hai nguyên nhân dễ gây "không gọi tool" nhất mà sửa rẻ; Giai đoạn 2 chỉ nên làm sau khi có raw thật.

## Rủi ro và cách xử lý

- **Heuristic retry sai (GĐ 4):** có thể retry thừa. Giới hạn 1 lần và có công tắc nên chi phí chấp nhận được.
- **Gemini web không tuân theo prompt dù đã mạnh:** là giới hạn của model. Retry là lớp bảo hiểm cuối, log cho thấy tần suất.
- **Cấu trúc raw khác giả định (GĐ 2):** vì vậy phải xem raw thật trước.
- **Cắt prompt vẫn mất ngữ cảnh:** system prompt opencode rất dài. Nếu vẫn tệ, cân nhắc nâng `PROMPT_MAX_BYTES` sau khi thử xem upstream còn nhận không.

---

## Prompt gọn giao cho model nhỏ

```
Repo: gemini-web2api (Python). Mục tiêu: tool calling qua /v1/chat/completions
phải ổn định với opencode. Làm TUẦN TỰ theo file KE_HOACH_SUA_TOOL_CALLING.md,
mỗi giai đoạn chạy `python -m unittest` xanh rồi mới sang giai đoạn sau.
Không refactor ngoài phạm vi. Không đổi API công khai. Mỗi thay đổi hành vi
có công tắc trong config.py. Mỗi giai đoạn một commit và có unit test mới
trong tests/test_modular_sync.py theo style hiện có (mock generate).
Bắt đầu với Giai đoạn 1, sau đó 3a, 3c, 4. Dừng và báo cáo trước Giai đoạn 2
(cần raw response thật từ Giai đoạn 0).
```
