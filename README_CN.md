# gemini-web2api

<p align="center">
  <img src="logo.png" width="200" alt="gemini-web2api logo">
</p>

[English](README.md)

将 Google Gemini 网页界面转换为兼容 OpenAI 的 API。零成本、跨平台、单文件。

## 功能

- **可选 API Key**: `api_keys` 为空时无需认证，配置后使用 OpenAI 风格 Bearer Key
- **OpenAI 兼容**: 可直接替换 `/v1/chat/completions` 与 `/v1/models`
- **工具调用**: 完整支持 Function Calling (OpenAI 格式)
- **多模型**: Flash (3.8)、扩展思考 (2万+字符)、Pro、Auto、Lite
- **思考深度**: 通过 `@think=N` 后缀调节 (0=最深, 4=最浅)
- **网页搜索**: 内置互联网访问 (Gemini 原生搜索)
- **跨平台**: 纯 Python，单一可选依赖 (`httpx` 用于流式)
- **流式传输**: 通过 `httpx` 支持 SSE Streaming
- **Codex CLI**: Responses API (`/v1/responses`) 兼容 OpenAI Codex
- **Gemini CLI**: Google 原生 API (`/v1beta/models`) 兼容 Gemini CLI

## 快速开始

```bash
pip install httpx
python gemini_web2api.py
```

服务启动于 `http://localhost:8081/v1`。

## 客户端配置

### Cherry Studio / ChatBox / 任意 OpenAI 客户端

| 字段 | 值 |
|------|-----|
| Base URL | `http://localhost:8081/v1` |
| API Key | `config.json` 中的 `api_keys` 值；未配置则任意值均可 |
| Model | `gemini-3.5-flash-thinking` |

### curl

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"你好！"}]}'
```

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")
resp = client.chat.completions.create(
    model="gemini-3.5-flash-thinking",
    messages=[{"role": "user", "content": "解释量子计算"}]
)
print(resp.choices[0].message.content)
```

### Gemini CLI

```bash
export GEMINI_API_KEY=none
export GOOGLE_GEMINI_BASE_URL=http://localhost:8081
gemini
```

支持 Google 原生 API 端点：
- `GET /v1beta/models` - 列出模型
- `POST /v1beta/models/{model}:generateContent` - 非流式
- `POST /v1beta/models/{model}:streamGenerateContent` - 流式 (SSE)

## 可用模型

| 模型 | 说明 | 输出 |
|------|------|--------|
| `gemini-3.8-flash` | 主力模型，推理与编程最强 (最新) | ~1.2万字符 |
| `gemini-3.8-flash-thinking` | 最新 Flash 后端的扩展思考 | **~2万字符** |
| `gemini-3.7-flash` | 全能模型 | ~1.2万字符 |
| `gemini-3.6-flash` | 全能模型 | ~1.2万字符 |
| `gemini-3.5-flash` | 全能模型 | ~1.2万字符 |
| `gemini-3.5-flash-thinking` | 扩展思考，最长输出 | **~2万字符** |
| `gemini-3.5-flash-thinking-lite` | 自适应思考深度 | ~1.5万字符 |
| `gemini-3.5-flash-lite` | 高性价比、大容量 | ~1万字符 |
| `gemini-3.1-flash-lite` | 高性价比、大容量 | ~1万字符 |
| `gemini-3.1-pro` | 高阶数学与代码 (需 cookie) | ~1.2万字符 |
| `gemini-auto` | 自动选择模型 | 不固定 |
| `gemini-flash-lite` | 最快响应，轻量级 | ~1万字符 |

### 思考深度

在任意模型名后追加 `@think=N`：

```
gemini-3.5-flash-thinking@think=0   # 最深 (默认)
gemini-3.5-flash-thinking@think=2   # 中等
gemini-3.5-flash-thinking@think=4   # 最浅
```

## 可选：Cookie 以启用 Pro

匿名访问对所有模型均有效，但 `gemini-3.1-pro` 无认证时会路由到 Flash。要获得真实 Pro 路由，需要 **Gemini Advanced (付费订阅)** 账号的 cookie：

```bash
python gemini_web2api.py --cookie-file cookie.txt
```

### 如何获取 Cookie

1. 打开 Chrome，访问 [gemini.google.com](https://gemini.google.com) 并用 **Gemini Advanced** 账号登录
2. 打开开发者工具 (F12) -> Application -> Cookies -> `https://gemini.google.com`
3. 复制以下 cookie 值：`SID`、`HSID`、`SSID`、`APISID`、`SAPISID`、`__Secure-1PSID`
4. 创建 `cookie.txt`，格式如下：

```
SID=你的SID值; HSID=你的HSID值; SSID=你的SSID值; APISID=你的APISID值; SAPISID=你的SAPISID值; __Secure-1PSID=你的1PSID值
```

或使用 JSON 格式：
```json
{"cookie": "SID=xxx; HSID=xxx; SSID=xxx; APISID=xxx; SAPISID=xxx; __Secure-1PSID=xxx", "sapisid": "你的SAPISID值"}
```

**替代方案 (浏览器扩展)**：使用任意 "Export Cookies" 扩展导出 `gemini.google.com` 的 Netscape 格式 cookie，再转为上述单行格式。

### 认证账号路径与 XSRF Token

若登录后的 Gemini 页面 URL 包含账号索引，例如：

```
https://gemini.google.com/u/1/app/...
```

则将 `auth_user` 设为该索引。已认证的网页请求可能还需要页面 XSRF token。在渲染后的 Gemini 页面源码中，该 token 以 `SNlM0e` 暴露；在 `config.json` 中作为 `xsrf_token` 传入，服务器会将其作为 `at` 表单字段发送。

示例：

```json
{
  "cookie_file": "/app/cookie.txt",
  "auth_user": "1",
  "xsrf_token": "AOOh0P...",
  "gemini_bl": "boq_assistant-bard-web-server_YYYYMMDD.xx_p0"
}
```

若认证请求返回 HTTP 400 并提示 `xsrf` 错误，请刷新 Gemini 网页，更新 `xsrf_token`，并确保 `auth_user` 与浏览器 URL 中 `/u/<索引>/` 部分一致。

Pro 路由需要 **Gemini Advanced (付费订阅)**。免费 Google 账号 cookie 只会认证通过，但静默回退到 Flash。

## 配置

在同目录下创建 `config.json`：

```json
{
  "port": 8081,
  "host": "0.0.0.0",
  "retry_attempts": 3,
  "retry_delay_sec": 2,
  "request_timeout_sec": 180,
  "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
  "auth_user": null,
  "xsrf_token": null,
  "api_keys": ["sk-your-key"],
  "cookie_file": null,
  "proxy": null,
  "log_requests": true,
  "temporary_chats": false
}
```

将 `temporary_chats` 设为 `true` 可使用 Gemini Web 的临时对话，而非持久化到账号历史。

`api_keys` 为 `[]` 时禁用认证；设置一个或多个 key 时，`/v1/*` 端点要求 `Authorization: Bearer <key>` 或 `x-api-key: <key>`。

## Docker

```bash
cp config.example.json config.json
docker build -t gemini-web2api .
docker run -d --name gemini-web2api -p 8081:8081 -v ./config.json:/app/config.json gemini-web2api
```

或使用 Docker Compose：

```bash
cp config.example.json config.json
docker compose up -d
```

挂载 cookie 文件：

```bash
docker run -d --name gemini-web2api -p 8081:8081 -v ./config.json:/app/config.json -v ./cookie.txt:/app/cookie.txt gemini-web2api
```

在 `config.json` 中设置 `"cookie_file": "/app/cookie.txt"`。

> **注意**：若 Docker 默认 bridge 网络下收到空响应 (`content: null`)，请改用 host 网络：`docker run --network host ...` 或 compose 中加 `network_mode: host`。这是 Gemini 上游拒绝某些 Docker NAT IP 段导致的。

## 代理

若无法直连 `gemini.google.com` (连接超时)，可配置代理：

**方式 1：命令行参数**
```bash
python gemini_web2api.py --proxy http://127.0.0.1:7890
```

**方式 2：config.json**
```json
{"proxy": "http://127.0.0.1:7890"}
```

**方式 3：环境变量** (自动检测)
```bash
set HTTPS_PROXY=http://127.0.0.1:7890
python gemini_web2api.py
```

兼容 Clash、V2Ray、Shadowsocks 或任意 HTTP 代理。

## 工具调用

Chat Completions 与 Responses API 均支持 OpenAI 格式多模态消息，可使用 HTTP(S) 图片 URL 或 base64 data URL：

```python
resp = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
    }]
)
```

## 图片输入

OpenAI 风格的多模态消息在 Chat Completions 与 Responses API 中均受支持。可使用 HTTP(S) 图片 URL 或 base64 data URL：

```python
resp = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
    }]
)
```

## 局限性

- **图片上传可能需要 Cookie**: 多模态输入使用 Gemini Web 的图片上传端点。若匿名上传失败，请配置 Gemini cookie。
- **非真实 Pro/Ultra**: 无付费订阅 cookie 时，`gemini-3.1-pro` 会路由到相同的 Flash 模型。"Pro" 标签仅为 UI 偏好，非后端模型切换。
- **仅单轮**: 每次请求为独立对话。通过在 prompt 中包含历史消息来模拟多轮上下文。
- **速率限制**: Google 可能会限制高频请求。服务器会自动重试，但持续大量使用仍可能被封禁。

## 依赖

- Python 3.8+
- `httpx` (`pip install httpx`) - 用于流式请求
- 可访问 `gemini.google.com` (部分地区需代理/VPN)

## 原理

本工具逆向工程了 Google Gemini 网页版的 StreamGenerate 协议。它向 Gemini 网页应用使用的同一端点发送请求，在 OpenAI API 格式与 Gemini 内部 protobuf-like 格式之间转换。

模型选择由请求 payload 中的字段 `[79]` 控制，对应 Gemini 前端 JavaScript 源码中的 `MODE_CATEGORY` 枚举。

## 致谢

- 受开源 API 代理生态启发

## License

MIT