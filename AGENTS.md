# AGENTS.md — gemini-web2api

> Onboarding doc for a new AI session: grasp the **algorithms, working
> principles, and logic** before reading the source. Language: English, keeping
> original English technical terms. This file is committed to git.

## 1. What the project is

**gemini-web2api** is a proxy that turns the Gemini **web frontend**
(`gemini.google.com`, internal endpoint `BardChatUi/.../StreamGenerate`) into
standard APIs:

- `POST /v1/chat/completions` (OpenAI, incl. streaming SSE + tool calling)
- `POST /v1/responses` (OpenAI Responses API for Codex CLI)
- `GET /v1/models`, `GET /v1beta/models`
- `POST /v1beta/models/{model}:generateContent` and `:streamGenerateContent`
  (Google native, for Gemini CLI)

No official Google API key is used — the proxy emulates a browser sending an
`application/x-www-form-urlencoded` request to StreamGenerate, then parses the
response.

## 2. Repo layout and MIRROR RULE (most important)

| Path | Role |
|---|---|
| `gemini_web2api/config.py` | `DEFAULT_CONFIG` + global `CONFIG` dict, `model_tickets`, config file loading |
| `gemini_web2api/models.py` | `MODELS`, `TICKET_HEADER`, `ticket_for()`, `resolve_model()` |
| `gemini_web2api/gemini.py` | StreamGenerate protocol: cookie/auth, payload+header building, `generate`/`generate_stream`, response parsing, echo/routing check |
| `gemini_web2api/server.py` | HTTP server (`GeminiHandler`): 3 endpoint families, SSE, image upload |
| `gemini_web2api/tools.py` | Tool calling: prompt injection (`messages_to_prompt`, `google_contents_to_prompt`), output parsing (`parse_tool_calls`, `parse_google_function_calls`) |
| `gemini_web2api/multimodal.py` | Image detect/fetch/upload |
| `gemini_web2api/__main__.py`, `__init__.py` | Entry point, version |
| **`gemini_web2api.py`** (root) | **Single-file mirror**: 1-file bundle of the whole package, for users running `python gemini_web2api.py` directly |
| `tests/test_openai_compat.py` | Parser + streaming tool_calls tests |
| `tests/test_modular_sync.py` | Package-vs-single-file sync, model mapping, ticket, echo tests |
| `cloudflare/worker.js` | Cloudflare Worker variant (independent, NOT part of the mirror) |
| `gemini-cookie-sync-extension/` | Cookie-sync extension (independent) |

**MIRROR RULE:** every change in `gemini_web2api/*.py` **must** be mirrored to
`gemini_web2api.py` (and vice versa). `test_modular_sync.py` checks this sync
— always run the full suite after each change. README/README_CN.md also need
their model tables synced when the model list changes.

**DOC RULE:** every change that alters behavior, mapping, ticket table, auth
flow, tool-call format, or working principles **must** update this `AGENTS.md`
(and README/README_CN if user-facing) in the same change — never leave docs
describing old code. Check: re-read the relevant section after editing source;
if any numbers/tables/function names in the docs diverge from code, fix
immediately.

## 3. Request flow (overview)

```
client → server.py (resolve_model + ticket_for)
       → gemini.generate / generate_stream (cookie + ticket header + payload)
       → POST gemini.google.com/.../StreamGenerate
       → extract_response_text (parse wrb.fr) + check_routing (echo)
       → server.py reformats per called API (OpenAI chunk / Responses event / Google candidate)
```

`resolve_model(name)` returns `(name, mode_id, think_mode, err, extra_fields)`:

- Supports `@think=N` suffix overriding thinking depth.
- Unknown model → **falls back** to `default_model`, no error (upstream
  clients may request arbitrary identifiers).
- `extra_fields` carries the `inner[80]` variant; `ticket_for()` looks up the
  ticket by key.

## 4. MODEL ROUTING algorithm (core, verified live)

Discovered via browser capture + 3/3 probe matrix (Sep 2026).
**The server routes BY TICKET, not by body.**

- Body `f.req` inner array: `inner[79]` = family, `inner[80]` = variant.
  Family: `1`=flash, `3`=pro, `5`=dynamic-thinking (thinking-lite's body),
  `6`=flash-lite. Variant: `1`=standard, `2`=extended/thinking.
- Header `X-Goog-Ext-525001261-Jspb` ("ticket", minted by the browser per model
  family) embeds **plaintext** `(family, variant)` at JSON indices `[14]`,
  `[15]`. **With a ticket, the ticket beats the body; without a ticket the
  server ignores `[79]`/`[80]` and falls back to the account default.**
- Ruled out (bisected one by one): `inner[3]`, `f.sid`, `at`, sibling headers
  `x-goog-ext-52500*` / `x-browser-*` — none affect routing.
- **No field selects the exact 3.x point version** within a family; the server
  picks its current default (e.g. there is no way to distinguish 3.5 vs 3.6
  Flash — the ticket only says "Flash").
- The proxy echoes `"model"` from the request name — that is a fake echo; the
  **only ground truth is the upstream echo** (the `wrb.fr` response contains
  `[label, ..., family, variant]` at indices `[42]`, `[58]`, `[59]`), read via
  `upstream_echo()`.

Current ticket table (`CONFIG["model_tickets"]`, key → `(fam,var)` in ticket):

| key | (fam,var) | used for |
|---|---|---|
| `flash` | (1,1) | plain flash |
| `flash-thinking` | (1,2) | flash extended |
| `lite` | (6,1) | flash-lite |
| `lite-thinking` | (6,2) | flash-lite extended (body sends fam 5, ticket wins → serves 6) |
| `pro` | (3,1) | pro |
| `pro-thinking` | (3,2) | pro extended |

`check_routing()` reads the expectation **from the ticket** (because the
ticket beats the body) and logs a `Routing mismatch` warning when upstream
serves something else — a sign of an expired ticket OR exhausted quota for
that family. Tickets embed a timestamp → **they will expire**; refresh by
copying the header from a fresh browser StreamGenerate request (DevTools →
Copy as cURL) into the matching key. Pro/pro-thinking and lite/lite-thinking
each share a key-id/UUID pair.

Current model list tracks the **web UI** (6 models, Sep 2026): Flash 3.6
(`3.6-flash`, `3.6-flash-thinking`), Lite 3.5 (`3.5-flash-lite`,
`3.5-flash-thinking-lite`), Pro 3.1 (`3.1-pro`, `3.1-pro-thinking`).
The `3.x` in the name is just a label — routing is decided by the ticket
`(family, variant)`; the server serves the family's current default. Unknown
names → fallback to `default_model`. Removed: `auto`, `3.8-flash`,
`3.8-flash-thinking`, `3.7-flash`, `pro-enhanced`, `3.5-flash` (old alias),
`3.1-flash-lite`, `gemini-flash-lite` (not on the UI). Thinking uses `think=1`
(browser default; `think=0` and `4` are deepest/shallowest per capture, with
longer/shorter outputs).

## 5. Cookie / Auth (avoiding 400s)

- `load_cookie()`: reads the cookie file (JSON `{cookie, sapisid, auth_user,
  xsrf_token, gemini_bl}` or a raw string), caches by mtime, **syncs
  `xsrf_token`/`gemini_bl`/`auth_user` into CONFIG**. A missing `at` param or
  stale `bl` → Gemini returns 400.
- `SAPISIDHASH`: `Authorization: SAPISIDHASH {ts}_{sha1(ts + sapisid + url)}`.
- Non-default `auth_user` → path prefix `/u/{n}` + `X-Goog-AuthUser` header.
- `refresh_auth()`: `SNlM0e` (xsrf) rotates every few minutes → statically
  exported tokens go stale fast. On 400-xsrf (detected via `_is_xsrf_error`)
  or when a cookie exists but no token yet: re-GET the `/app` page with the
  cookie session, extract `SNlM0e` + `bl`, persist back into the JSON cookie
  file, retry the request once.
- `generate_stream` uses httpx (prefix-based delta-diff, guards against
  content shifts between retries); falls back to `generate` without httpx.

## 6. Tool calling (prompt-injection style, since the web frontend has no tool field)

- Request: `messages_to_prompt` / `google_contents_to_prompt` inject the tool
  spec into the prompt as markdown + force the output format
  (`` ```tool_call `` / `` ```function_call ``) + constrain per `tool_choice`
  (none/auto/required/specific name; Google: NONE/ANY + allowedFunctionNames).
- Response: `parse_tool_calls` accepts 5 formats (tool_call fence,
  function_call fence, json fence, `[tool_call: name {...}]`, raw JSON);
  unparseable fences (e.g. code samples) are kept as-is; unknown tool names
  are filtered via `valid_names`.
- Streaming OpenAI-spec (`_stream_tool_calls`): role chunk → per-call id/name
  **with `index`** → argument slices (120 chars) → `tool_calls` finish →
  `[DONE]`. A missing `index` was an old bug that stopped AI SDKs from
  assembling calls.

## 7. Working principles (distilled from sessions)

1. **Evidence before synthesis**: every upstream claim (routing, which field
   decides) must be verified with a **live probe** (script sending a real
   StreamGenerate request, reading `upstream_echo`), never guessed. A full
   matrix (with/without ticket × varying bodies) is needed before concluding.
2. **Bisect on suspicion**: disable headers/fields one at a time to find the
   real cause (this is how the ticket header was found).
3. **Mirror + test after every change**: `py -m py_compile` both copies +
   `py -m unittest discover -s tests` (35 tests). A `Routing mismatch` log
   line during manual testing is a normal diagnostic signal, not a proxy bug.
4. **Distinguish quota vs mapping**: a whole model family downgrading at once
   (even a previously good ticket drops) → account quota exhausted, not a
   mapping bug.
5. **Environment**: use the `py` launcher (`python`/`python3` are not on
   PATH); PowerShell (no `&&` chaining, careful quoting); temp scripts go to
   `%TEMP%\opencode`, deleted after use.
6. **Git**: branch `fix/openai-tool-calling-compat` on fork
   `vchieu/gemini-web2api`, PR #99 (cookie sync, minimal) and #100 (superset)
   to `Sophomoresty/gemini-web2api`. Never commit local files
   (`gemini-auth.json`, `config.json`, ...).

## 8. How to refresh an expired model ticket (standard procedure)

1. Open `gemini.google.com`, select the right model in the UI, send one message.
2. DevTools → Network → the `StreamGenerate` request → Copy as cURL.
3. Take exactly **1 header**: `-H 'x-goog-ext-525001261-jspb: [...]'`.
4. Paste it into the matching key in `CONFIG["model_tickets"]`
   (`config.py` + mirror `gemini_web2api.py`), check `[14]`/`[15]` yield the
   right `(fam,var)`, verify live with an echo probe, run tests, commit.
