# LLM Transform Proxy

An OpenAI-compatible proxy that sits between your AI coding tools (bolt.diy, Cursor, Zed, etc.) and llama.cpp. It intercepts responses, detects malformed tool calls, and **transforms them into proper format**.

## The Problem This Solves

When using Qwen3-Coder and similar models with agentic coding tools, the model sometimes produces malformed tool calls like:

```
I'll help you create that file.
<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("hello")</parameter>
</function>
```

When it should produce a response with proper `tool_calls` JSON:

```json
{
  "choices": [{
    "message": {
      "content": "I'll help you create that file.",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "writeFile",
          "arguments": "{\"path\":\"src/app.js\",\"content\":\"console.log(\\\"hello\\\")\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

The **transform proxy** parses the malformed XML and reconstructs it as proper OpenAI-compatible tool calls.

---

## Quick Start

### Option 1: Using Nix (Recommended)

If you have Nix with flakes enabled:

```bash
# Enter the development shell
nix develop

# Or with direnv (automatic on cd)
direnv allow

# Run the proxy
python transform_proxy.py --backend http://localhost:8080 --port 4000
```

Or run directly without entering the shell:

```bash
# Run transform proxy
nix run .#transform -- --backend http://localhost:8080 --port 4000
```

### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the proxy
python transform_proxy.py --backend http://localhost:8080 --port 4000
```

### Then...

```bash
# Start llama.cpp server (in another terminal)
llama-server -m your-model.gguf -c 16384 --jinja --port 8080

# Configure your client to use http://localhost:4000 as the API base URL
```

---

## Project Structure

```
toolbridge/
├── flake.nix              # Nix flake for reproducible environment
├── .envrc                 # direnv configuration (auto-activates nix shell)
├── .gitignore
├── requirements.txt       # pip dependencies (alternative to nix)
├── transform_proxy.py     # Parse & transform proxy
├── test_transform.py      # Test suite for transformation logic
├── toolbridge.service      # Systemd unit (system-wide)
├── toolbridge.user.service # Systemd unit (user-level, no root)
└── README.md
```

### Nix Flake Features

The `flake.nix` provides:

| Command | Description |
|---------|-------------|
| `nix develop` | Enter dev shell with Python + all deps |
| `nix run .#transform` | Run transform proxy (single worker) |
| `nix run .#production` | Run with gunicorn (4 workers default) |
| `nix build` | Build the package |

Environment variables for production:
```bash
WORKERS=4 PORT=4000 HOST=0.0.0.0 nix run .#production
```

The dev shell includes:
- Python 3.12
- fastapi, uvicorn, httpx
- Development tools: pytest, black, ruff, mypy
- CLI utilities: curl, jq

---

## How Transformation Works

The proxy parses various malformed formats:

```
Input formats supported:
├── <function=name>...</function>           (most common)
├── <function="name">...</function>         (quoted)
├── <function name="name">...</function>    (attribute style)
└── {"name": "...", "arguments": {...}}     (raw JSON)

Output format:
└── OpenAI tool_calls JSON structure
```

### Transformation Example

**Input (malformed):**
```
I'll create that file for you.
<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("Hello!");</parameter>
</function>
```

**Output (proper OpenAI format):**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'll create that file for you.",
      "tool_calls": [{
        "id": "call_9406505917684901935ce4ee",
        "type": "function",
        "function": {
          "name": "writeFile",
          "arguments": "{\"path\": \"src/app.js\", \"content\": \"console.log(\\\"Hello!\\\");\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Test the Transformation

The proxy includes a test endpoint:

```bash
curl "http://localhost:4000/proxy/test-transform"
```

Or with custom content:

```bash
curl "http://localhost:4000/proxy/test-transform?content=<function=test><parameter=x>1</parameter></function>"
```

---

## Concurrency & Scaling

### Default Mode (Single Worker, Async I/O)

The proxy uses Python's `asyncio` for concurrent request handling:

```bash
python transform_proxy.py --port 4000
```

This handles multiple concurrent requests via async I/O. While one request waits for the llama.cpp backend, others can proceed. The transformation logic is fast (~microseconds) and won't bottleneck.

**Good for:** Single user, development, or when llama.cpp is the bottleneck anyway.

### Multi-Worker Mode (Production)

For multiple concurrent users or higher throughput:

```bash
# Using uvicorn directly with multiple workers
python -m uvicorn transform_proxy:app --workers 4 --host 0.0.0.0 --port 4000

# Or using gunicorn (more production-ready)
gunicorn transform_proxy:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:4000
```

| Mode | Workers | Concurrent Requests | Use Case |
|------|---------|---------------------|----------|
| Default | 1 (async) | Many (I/O bound) | Dev, single user |
| `--workers 4` | 4 processes | 4x throughput | Multiple users |
| gunicorn | N processes | Production scale | Team/production |

### Thread Safety

The proxy has **no shared mutable state** - each request is independent. Multi-worker mode is safe with no additional configuration.

### Bottleneck Analysis

```
Request → Proxy (fast) → llama.cpp (slow) → Proxy (fast) → Response
              │                  │                │
          ~1-5ms            1-60 sec          ~1-5ms
```

In practice, llama.cpp is almost always the bottleneck. Multiple proxy workers only help if:
- llama.cpp has multiple slots (`-np 2` or higher)
- You're running multiple llama.cpp instances
- Transformation parsing becomes measurable (very large responses)

---

## Understanding Streaming vs Non-Streaming

### What is Streaming?

When you make an API request to an LLM, you can choose between two modes:

| Mode | How it works | User experience |
|------|--------------|-----------------|
| **Non-streaming** | Server generates entire response, sends it all at once | User waits, then sees complete response |
| **Streaming** | Server sends tokens as they're generated via SSE | User sees response appear word-by-word |

### How is Streaming Controlled?

Streaming is controlled by the `stream` parameter in the API request:

```json
// Non-streaming request
{
  "model": "qwen3-coder",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}

// Streaming request
{
  "model": "qwen3-coder", 
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

### Who Controls Streaming?

**The client application controls streaming, not you directly.** Here's where different tools set this:

| Tool | Streaming Default | How to Change |
|------|-------------------|---------------|
| **bolt.diy** | `true` (streaming) | Check settings, may have toggle |
| **Cursor** | `true` (streaming) | Usually not configurable |
| **OpenCode** | `true` (streaming) | `--no-stream` flag or config |
| **Zed** | `true` (streaming) | Check assistant settings |
| **cURL/scripts** | Your choice | Set `"stream": false` in request |

### Why Streaming Complicates Retry Logic

With non-streaming:
```
Client → Proxy → Backend
                    ↓
             Generate full response
                    ↓
             Return to proxy
                    ↓
         Proxy validates response
                    ↓
    If malformed: retry with new request
                    ↓
         Return valid response to client
```

With streaming:
```
Client → Proxy → Backend
                    ↓
             Start generating tokens
                    ↓
    Token 1 → Token 2 → Token 3 → ... → Token N
         ↑
    Problem: Can't validate until we see the whole response!
```

### Proxy Streaming Strategies

This proxy implements **buffer-and-replay** for streaming:

```
1. Client requests streaming response
2. Proxy intercepts and collects ALL tokens from backend
3. Once complete, proxy validates the full response
4. If valid: replay collected tokens to client (appears as stream)
5. If invalid: retry with backend, then replay successful response
```

**Tradeoff:** This adds latency equal to the full generation time. The client won't see any tokens until the entire response is generated and validated.

#### Alternative Strategies (Not Implemented)

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Passthrough** | Stream directly, no validation | No latency | No retry capability |
| **Early abort** | Detect malformed pattern mid-stream, abort, retry | Some latency savings | Complex, may false positive |
| **Hybrid** | Stream until tool call detected, then buffer | Best of both | Very complex |

---

## Configuration

### Transform Proxy Command Line Options

```bash
python transform_proxy.py [OPTIONS]

Connection:
  --backend, -b URL     Backend llama.cpp server URL (default: http://localhost:8080)
  --port, -p PORT       Port to listen on (default: 4000)
  --host HOST           Host to bind to (default: 0.0.0.0)

Behavior:
  --no-transform        Disable transformation (passthrough mode)
  --debug               Enable debug logging
  --no-log-params       Disable logging of sampling parameters

Sampling Overrides (optional - override client values):
  --temperature FLOAT   Override temperature (e.g., 0.7)
  --top-p FLOAT         Override top_p (e.g., 0.9)
  --top-k INT           Override top_k (e.g., 40)
  --min-p FLOAT         Override min_p (e.g., 0.05)
  --repeat-penalty FLOAT Override repeat_penalty (e.g., 1.0)
  --presence-penalty FLOAT Override presence_penalty
  --frequency-penalty FLOAT Override frequency_penalty
```

### Sampling Parameter Overrides

When you specify a sampling parameter at the proxy level, it **overrides** whatever the client sends. If not specified, client values pass through unchanged to the backend.

```bash
# Force specific temperature and repeat_penalty for all requests
python transform_proxy.py --temperature 0.7 --repeat-penalty 1.0

# Just observe what clients send (no overrides)
python transform_proxy.py --debug
```

**Log output shows what's happening:**
```
Sampling params - Client: {'temperature': 0.6}, Proxy overrides: {'temperature': 0.7}, Effective: {'temperature': 0.7}
```

### Runtime Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config` | GET | View current configuration including sampling overrides |
| `/stats` | GET | View statistics and current sampling config |
| `/stats/reset` | POST | Reset statistics counters |
| `/health` | GET | Health check |

```bash
# Check current config
curl -s http://localhost:4000/config | jq

# Check stats and see what overrides are active
curl -s http://localhost:4000/stats | jq '.sampling_overrides'
```

### Examples

```bash
# Basic usage - transform proxy
python transform_proxy.py

# Custom backend and port
python transform_proxy.py --backend http://192.168.1.100:8080 --port 5000

# With sampling overrides
python transform_proxy.py --temperature 0.7 --repeat-penalty 1.0 --top-p 0.9
```

---

## Client Configuration

### bolt.diy

In your `.env` or settings:

```env
OPENAI_API_BASE_URL=http://localhost:4000/v1
# or
OLLAMA_API_BASE_URL=http://localhost:4000
```

Or in the UI, set the API endpoint to `http://localhost:4000`.

### Cursor

In Settings → Models → OpenAI API Base:

```
http://localhost:4000/v1
```

### Zed

In `settings.json`:

```json
{
  "language_models": {
    "openai": {
      "api_url": "http://localhost:4000"
    }
  }
}
```

### OpenCode

```bash
opencode --api-base http://localhost:4000/v1 --model local-model
```

### Direct API Calls

```bash
# Non-streaming (recommended for testing)
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "messages": [{"role": "user", "content": "Create a hello world file"}],
    "stream": false
  }'

# Streaming
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder", 
    "messages": [{"role": "user", "content": "Create a hello world file"}],
    "stream": true
  }'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Machine                            │
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐ │
│  │   Client    │      │  Transform  │      │   llama.cpp     │ │
│  │  (bolt.diy) │ ───▶ │   Proxy     │ ───▶ │    Server       │ │
│  │             │      │  :4000      │      │    :8080        │ │
│  └─────────────┘      └─────────────┘      └─────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│                     ┌───────────────┐                          │
│                     │    Parse &    │                          │
│                     │   Transform   │                          │
│                     └───────┬───────┘                          │
│                             │                                  │
│                             ▼                                  │
│                      Return to client                          │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Main endpoint with retry logic |
| `/v1/models` | GET | Proxied to backend |
| `/health` | GET | Proxy health check |
| `/proxy/stats` | GET | Proxy statistics |
| `/*` | * | All other endpoints passed through |

---

## Monitoring

### Health Check

```bash
curl http://localhost:4000/health
```

Response:
```json
{
  "status": "healthy",
  "backend_url": "http://localhost:8080",
  "backend_healthy": true
}
```

### Logs

The proxy logs transformation activity:

```
2026-01-20 10:30:15 - INFO - Received request: streaming=True, messages=5
2026-01-20 10:30:18 - INFO - Transformed malformed tool call to proper format
2026-01-20 10:30:18 - INFO - Returning response to client
```

---

## Performance Considerations

### Latency Impact

| Scenario | Additional Latency |
|----------|-------------------|
| Valid response, non-streaming | ~0ms (just passthrough) |
| Valid response, streaming | Full generation time (buffer-and-replay) |
| Transformation needed | ~0ms (parsing is fast) |

### Memory Usage

For streaming with buffer-and-replay, the entire response is held in memory. For typical coding tasks (1-10KB responses), this is negligible.

### When to Use This vs Fix the Template

This proxy is a **workaround**, not a permanent solution. Consider:

- Use the proxy when you can't modify the model's chat template
- If you have control over the template, fix it at the source
- Our optimized chat template achieved 97-100% success rates, potentially making this proxy unnecessary

---

## Systemd Service Installation

### User Service (No Root Required)

This runs the proxy as your user, starts on login:

```bash
# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Copy the service file
cp toolbridge.user.service ~/.config/systemd/user/toolbridge.service

# Edit to match your setup (paths, backend URL, etc.)
nano ~/.config/systemd/user/toolbridge.service

# Reload systemd
systemctl --user daemon-reload

# Enable (start on login) and start
systemctl --user enable --now toolbridge

# Check status
systemctl --user status toolbridge

# View logs
journalctl --user -u toolbridge -f
```

### System Service (Runs as Dedicated User)

For always-on servers or multi-user setups:

```bash
# Install the proxy files
sudo mkdir -p /opt/toolbridge
sudo cp transform_proxy.py /opt/toolbridge/
sudo cp requirements.txt /opt/toolbridge/

# Install Python deps (or use nix)
sudo pip install -r /opt/toolbridge/requirements.txt

# Install the service
sudo cp toolbridge.service /etc/systemd/system/

# Edit configuration
sudo nano /etc/systemd/system/toolbridge.service

# Reload, enable, start
sudo systemctl daemon-reload
sudo systemctl enable --now toolbridge

# Check status
sudo systemctl status toolbridge

# View logs
sudo journalctl -u toolbridge -f
```

### Systemd Environment Configuration

Edit the `Environment=` lines in the service file:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOLBRIDGE_BACKEND` | `http://localhost:8080` | llama.cpp server URL |
| `TOOLBRIDGE_PORT` | `4000` | Port to listen on |
| `TOOLBRIDGE_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for all interfaces) |

Or use an environment file:

```bash
# /etc/toolbridge/config (system) or ~/.config/toolbridge/config (user)
TOOLBRIDGE_BACKEND=http://localhost:8080
TOOLBRIDGE_PORT=4000
TOOLBRIDGE_HOST=127.0.0.1
```

Then uncomment `EnvironmentFile=` in the service file.

### Service Management Commands

```bash
# User service
systemctl --user start toolbridge
systemctl --user stop toolbridge
systemctl --user restart toolbridge
systemctl --user status toolbridge
journalctl --user -u toolbridge -f

# System service (add sudo)
sudo systemctl start toolbridge
sudo systemctl stop toolbridge
sudo systemctl restart toolbridge
sudo systemctl status toolbridge
sudo journalctl -u toolbridge -f
```

---

## Troubleshooting

### Proxy won't start

```bash
# Check if port is in use
lsof -i :4000

# Try a different port
python proxy.py --port 5000
```

### Backend connection refused

```bash
# Verify llama.cpp is running
curl http://localhost:8080/health

# Check backend URL
python proxy.py --backend http://127.0.0.1:8080
```

### Still getting malformed responses

1. Enable debug logging: `python transform_proxy.py --debug`
2. Check if your malformed pattern is being detected and transformed
3. The proxy supports various malformed formats - check the logs to see what's being parsed

### High latency with streaming

This is expected with buffer-and-replay. Options:
- Accept the latency (validation is worth it)
- Configure client to use non-streaming if possible
- Disable proxy for this client and rely on template fixes

---

## Extending the Proxy

### Metrics Collection

The `/stats` endpoint provides counters for:
- Total requests
- Transformation count
- Average latency
