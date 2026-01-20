#!/usr/bin/env python3
"""
LLM Retry Proxy - OpenAI-compatible proxy for tool call validation and retry

This proxy sits between your client (bolt.diy, Cursor, etc.) and llama.cpp server.
It intercepts responses and retries if malformed tool calls are detected.

Usage:
    python proxy.py --backend http://localhost:8080 --port 4000

Client configuration:
    Point your client's API base URL to http://localhost:4000
"""

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProxyConfig:
    """Proxy configuration settings"""
    backend_url: str = "http://localhost:8080"
    port: int = 4000
    host: str = "0.0.0.0"
    
    # Retry settings
    max_retries: int = 2
    retry_delay: float = 0.5  # seconds between retries
    
    # Detection patterns for malformed tool calls
    # These patterns indicate the model produced a function call without proper wrapping
    malformed_patterns: list = field(default_factory=lambda: [
        r'<function\s*=',           # <function=name>
        r'<function\s+name\s*=',    # <function name="...">
        r'<function_call>',          # Wrong tag name
        r'\{"name":\s*"[^"]+",\s*"arguments"',  # Raw JSON tool call without wrapper
    ])
    
    # Pattern that indicates PROPERLY formatted tool call
    valid_wrapper_pattern: str = r'<tool_call>'
    
    # Temperature adjustment on retry (lower = more deterministic)
    retry_temperature_adjustment: float = -0.1
    min_temperature: float = 0.3
    
    # Whether to add reminder prompts on retry
    add_retry_hints: bool = True
    
    # Streaming settings
    stream_buffer_timeout: float = 30.0  # Max time to buffer stream before giving up


config = ProxyConfig()
app = FastAPI(title="LLM Retry Proxy")

# =============================================================================
# Detection Logic
# =============================================================================

def detect_malformed_tool_call(content: str) -> tuple[bool, str]:
    """
    Detect if content contains a malformed tool call.
    
    Returns:
        (is_malformed, reason)
    """
    if not content:
        return False, ""
    
    # Check for malformed patterns
    for pattern in config.malformed_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            # But only if it's NOT properly wrapped
            if not re.search(config.valid_wrapper_pattern, content, re.IGNORECASE):
                return True, f"Found malformed pattern '{pattern}' without valid wrapper"
    
    # Additional heuristic: function-like content at end of response without wrapper
    # This catches cases where model writes preamble then jumps to <function=...
    if re.search(r'<function[^>]*>\s*$', content):
        if '<tool_call>' not in content:
            return True, "Response ends with <function> tag but no <tool_call> wrapper"
    
    return False, ""


def detect_incomplete_tool_call(content: str) -> bool:
    """
    Detect if a tool call was started but not completed.
    Useful for streaming to detect early and potentially abort.
    """
    # Has opening but no closing
    if '<tool_call>' in content and '</tool_call>' not in content:
        return True
    if '<function=' in content and '</function>' not in content:
        return True
    return False

# =============================================================================
# Message Augmentation for Retries
# =============================================================================

RETRY_HINT = """
<IMPORTANT>
Your previous response had a formatting error. When making function/tool calls:
1. ALWAYS wrap calls in <tool_call>...</tool_call> tags
2. The format is: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
3. Never output <function=...> without the <tool_call> wrapper
</IMPORTANT>
"""

def augment_messages_for_retry(
    messages: list[dict],
    failed_response: str,
    attempt: int
) -> list[dict]:
    """
    Augment messages with retry hints to help model correct its output.
    """
    messages = [m.copy() for m in messages]  # Deep copy
    
    if not config.add_retry_hints:
        return messages
    
    # Strategy 1: Add hint to system message
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + RETRY_HINT
    else:
        messages.insert(0, {"role": "system", "content": RETRY_HINT.strip()})
    
    # Strategy 2: On second retry, also show what went wrong
    if attempt >= 2 and failed_response:
        # Truncate the failed response to show the problematic part
        snippet = failed_response[-500:] if len(failed_response) > 500 else failed_response
        correction_msg = {
            "role": "user",
            "content": f"Your previous response had this formatting issue:\n```\n{snippet}\n```\nPlease try again with proper <tool_call> wrapping."
        }
        messages.append(correction_msg)
    
    return messages


def adjust_parameters_for_retry(body: dict, attempt: int) -> dict:
    """
    Adjust generation parameters for retry attempts.
    Lower temperature can help with format compliance.
    """
    body = body.copy()
    
    current_temp = body.get("temperature", 0.7)
    new_temp = max(
        config.min_temperature,
        current_temp + (config.retry_temperature_adjustment * attempt)
    )
    body["temperature"] = new_temp
    
    logger.info(f"Retry {attempt}: Adjusted temperature {current_temp} -> {new_temp}")
    
    return body

# =============================================================================
# Non-Streaming Handler
# =============================================================================

async def handle_non_streaming_request(
    client: httpx.AsyncClient,
    body: dict,
    original_messages: list[dict]
) -> dict:
    """
    Handle a non-streaming chat completion request with retry logic.
    """
    messages = original_messages.copy()
    last_response = None
    last_content = ""
    
    for attempt in range(config.max_retries + 1):
        request_body = body.copy()
        request_body["messages"] = messages
        
        if attempt > 0:
            request_body = adjust_parameters_for_retry(request_body, attempt)
            await asyncio.sleep(config.retry_delay)
        
        logger.info(f"Attempt {attempt + 1}/{config.max_retries + 1}: Sending request to backend")
        
        try:
            response = await client.post(
                f"{config.backend_url}/v1/chat/completions",
                json=request_body,
                timeout=120.0
            )
            response.raise_for_status()
            result = response.json()
            last_response = result
            
            # Extract content from response
            content = ""
            if "choices" in result and result["choices"]:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
            
            last_content = content
            
            # Check for malformed tool call
            is_malformed, reason = detect_malformed_tool_call(content)
            
            if not is_malformed:
                logger.info(f"Attempt {attempt + 1}: Response valid, returning")
                return result
            
            logger.warning(f"Attempt {attempt + 1}: Malformed tool call detected - {reason}")
            
            # Prepare for retry
            if attempt < config.max_retries:
                messages = augment_messages_for_retry(
                    original_messages, 
                    content, 
                    attempt + 1
                )
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from backend: {e}")
            raise
        except Exception as e:
            logger.error(f"Error communicating with backend: {e}")
            raise
    
    # All retries exhausted, return last response
    logger.warning(f"All {config.max_retries + 1} attempts failed, returning last response")
    return last_response

# =============================================================================
# Streaming Handler
# =============================================================================

async def collect_stream_content(
    client: httpx.AsyncClient,
    body: dict
) -> tuple[str, list[dict], dict]:
    """
    Collect entire stream into content string and list of chunks.
    Returns (full_content, all_chunks, final_metadata)
    """
    full_content = ""
    chunks = []
    final_metadata = {}
    
    async with client.stream(
        "POST",
        f"{config.backend_url}/v1/chat/completions",
        json=body,
        timeout=config.stream_buffer_timeout
    ) as response:
        response.raise_for_status()
        
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            
            data = line[6:]  # Remove "data: " prefix
            
            if data == "[DONE]":
                chunks.append({"done": True})
                break
            
            try:
                chunk = json.loads(data)
                chunks.append(chunk)
                
                # Extract content delta
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
                    
                    # Capture finish reason
                    if chunk["choices"][0].get("finish_reason"):
                        final_metadata["finish_reason"] = chunk["choices"][0]["finish_reason"]
                
                # Capture model info
                if "model" in chunk:
                    final_metadata["model"] = chunk["model"]
                if "id" in chunk:
                    final_metadata["id"] = chunk["id"]
                    
            except json.JSONDecodeError:
                continue
    
    return full_content, chunks, final_metadata


async def replay_chunks_as_stream(chunks: list[dict]) -> AsyncGenerator[str, None]:
    """
    Replay collected chunks as SSE stream.
    """
    for chunk in chunks:
        if chunk.get("done"):
            yield "data: [DONE]\n\n"
        else:
            yield f"data: {json.dumps(chunk)}\n\n"


async def generate_synthetic_stream(
    content: str,
    metadata: dict,
    chunk_size: int = 20
) -> AsyncGenerator[str, None]:
    """
    Generate a synthetic SSE stream from complete content.
    Used when we need to retry and replay.
    """
    chunk_id = metadata.get("id", f"chatcmpl-{int(time.time())}")
    model = metadata.get("model", "unknown")
    
    # Split content into chunks
    for i in range(0, len(content), chunk_size):
        chunk_content = content[i:i + chunk_size]
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_content},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)  # Small delay for realistic streaming
    
    # Final chunk with finish reason
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": metadata.get("finish_reason", "stop")
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def handle_streaming_request(
    client: httpx.AsyncClient,
    body: dict,
    original_messages: list[dict]
) -> AsyncGenerator[str, None]:
    """
    Handle a streaming chat completion request with retry logic.
    
    Strategy: Buffer the entire stream, validate, retry if needed,
    then replay to client.
    
    Note: This introduces latency since we can't stream until validation.
    For true streaming with early abort, more complex logic is needed.
    """
    messages = original_messages.copy()
    
    for attempt in range(config.max_retries + 1):
        request_body = body.copy()
        request_body["messages"] = messages
        request_body["stream"] = True
        
        if attempt > 0:
            request_body = adjust_parameters_for_retry(request_body, attempt)
            await asyncio.sleep(config.retry_delay)
        
        logger.info(f"Stream attempt {attempt + 1}/{config.max_retries + 1}")
        
        try:
            content, chunks, metadata = await collect_stream_content(client, request_body)
            
            # Validate collected content
            is_malformed, reason = detect_malformed_tool_call(content)
            
            if not is_malformed:
                logger.info(f"Stream attempt {attempt + 1}: Valid, replaying to client")
                async for chunk in replay_chunks_as_stream(chunks):
                    yield chunk
                return
            
            logger.warning(f"Stream attempt {attempt + 1}: Malformed - {reason}")
            
            # Prepare for retry
            if attempt < config.max_retries:
                messages = augment_messages_for_retry(
                    original_messages,
                    content,
                    attempt + 1
                )
                
        except Exception as e:
            logger.error(f"Stream error: {e}")
            if attempt >= config.max_retries:
                raise
    
    # All retries failed, replay last collected stream
    logger.warning("All stream attempts failed, replaying last response")
    async for chunk in replay_chunks_as_stream(chunks):
        yield chunk

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint with retry logic.
    """
    body = await request.json()
    original_messages = body.get("messages", [])
    is_streaming = body.get("stream", False)
    
    logger.info(f"Received request: streaming={is_streaming}, messages={len(original_messages)}")
    
    async with httpx.AsyncClient() as client:
        if is_streaming:
            return StreamingResponse(
                handle_streaming_request(client, body, original_messages),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            result = await handle_non_streaming_request(client, body, original_messages)
            return result


@app.get("/v1/models")
async def list_models():
    """Proxy models endpoint to backend."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{config.backend_url}/v1/models")
        return response.json()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.backend_url}/health",
                timeout=5.0
            )
            backend_healthy = response.status_code == 200
    except Exception:
        backend_healthy = False
    
    return {
        "status": "healthy" if backend_healthy else "degraded",
        "backend_url": config.backend_url,
        "backend_healthy": backend_healthy,
        "config": {
            "max_retries": config.max_retries,
            "add_retry_hints": config.add_retry_hints,
        }
    }


@app.get("/proxy/stats")
async def proxy_stats():
    """Proxy statistics (placeholder for future metrics)."""
    return {
        "message": "Stats endpoint - implement metrics collection as needed",
        "config": {
            "backend_url": config.backend_url,
            "max_retries": config.max_retries,
            "retry_delay": config.retry_delay,
            "malformed_patterns": config.malformed_patterns,
        }
    }

# =============================================================================
# Passthrough for other endpoints
# =============================================================================

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str):
    """
    Pass through any other requests to the backend unchanged.
    This ensures compatibility with all llama.cpp endpoints.
    """
    async with httpx.AsyncClient() as client:
        # Build the target URL
        url = f"{config.backend_url}/{path}"
        
        # Forward the request
        response = await client.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            content=await request.body() if request.method in ["POST", "PUT", "PATCH"] else None,
            params=request.query_params,
            timeout=120.0
        )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Retry Proxy")
    parser.add_argument(
        "--backend", "-b",
        default="http://localhost:8080",
        help="Backend llama.cpp server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=4000,
        help="Port to listen on (default: 4000)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=2,
        help="Maximum retry attempts (default: 2)"
    )
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="Disable retry hint injection"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Update config
    config.backend_url = args.backend
    config.port = args.port
    config.host = args.host
    config.max_retries = args.max_retries
    config.add_retry_hints = not args.no_hints
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting LLM Retry Proxy")
    logger.info(f"  Backend: {config.backend_url}")
    logger.info(f"  Listening: {config.host}:{config.port}")
    logger.info(f"  Max retries: {config.max_retries}")
    logger.info(f"  Retry hints: {config.add_retry_hints}")
    
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
