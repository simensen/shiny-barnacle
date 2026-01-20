#!/usr/bin/env python3
"""
LLM Response Transformer Proxy

An OpenAI-compatible proxy that transforms malformed tool calls into proper format.
Instead of retrying failed responses, it parses and reconstructs them.

Handles transformations like:
    <function=writeFile>                    →  tool_calls: [{
    <parameter=path>src/app.js</parameter>        function: {
    </function>                                      name: "writeFile",
                                                     arguments: '{"path":"src/app.js"}'
                                                   }
                                                 }]
"""

import argparse
import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Any
from xml.sax.saxutils import unescape as xml_unescape

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Stats Tracking
# =============================================================================

@dataclass
class ProxyStats:
    """Track proxy statistics for observability"""
    total_requests: int = 0
    passthrough_requests: int = 0  # Model output was already valid
    transformed_requests: int = 0  # Proxy fixed malformed output
    failed_transforms: int = 0     # Transformation attempted but failed
    backend_errors: int = 0        # Backend returned error
    
    def record_passthrough(self):
        self.total_requests += 1
        self.passthrough_requests += 1
        
    def record_transformed(self):
        self.total_requests += 1
        self.transformed_requests += 1
        
    def record_failed(self):
        self.total_requests += 1
        self.failed_transforms += 1
        
    def record_backend_error(self):
        self.total_requests += 1
        self.backend_errors += 1
        
    def to_dict(self) -> dict:
        total = self.total_requests or 1  # Avoid division by zero
        return {
            "total_requests": self.total_requests,
            "passthrough": {
                "count": self.passthrough_requests,
                "percent": round(100 * self.passthrough_requests / total, 1)
            },
            "transformed": {
                "count": self.transformed_requests,
                "percent": round(100 * self.transformed_requests / total, 1)
            },
            "failed": {
                "count": self.failed_transforms,
                "percent": round(100 * self.failed_transforms / total, 1)
            },
            "backend_errors": {
                "count": self.backend_errors,
                "percent": round(100 * self.backend_errors / total, 1)
            }
        }

# Global stats instance
stats = ProxyStats()

@dataclass
class ProxyConfig:
    """Proxy configuration settings"""
    backend_url: str = "http://localhost:8080"
    port: int = 4000
    host: str = "0.0.0.0"
    
    # Transformation settings
    transform_enabled: bool = True
    fallback_to_retry: bool = True  # If transform fails, try retry
    max_retries: int = 1  # Only used if transform fails
    
    # Streaming settings
    stream_buffer_timeout: float = 120.0


config = ProxyConfig()
app = FastAPI(title="LLM Response Transformer Proxy")

# =============================================================================
# Tool Call Parser
# =============================================================================

@dataclass
class ParsedToolCall:
    """Represents a parsed tool call"""
    function_name: str
    arguments: dict
    raw_text: str  # Original text that was parsed

@dataclass 
class ParsedResponse:
    """Represents a fully parsed response"""
    preamble: str  # Text before tool calls
    tool_calls: list[ParsedToolCall]
    postamble: str  # Text after tool calls (rare)
    was_transformed: bool


class ToolCallParser:
    """
    Parser for various malformed tool call formats.
    
    Supports:
    - <function=name>...</function>
    - <function name="name">...</function>
    - <tool_call><function=name>...</function></tool_call> (passthrough)
    - Raw JSON tool calls
    """
    
    # Pattern to match various function call formats:
    # - <function=name>...</function>
    # - <function="name">...</function>  
    # - <function name="name">...</function>
    # - <function name=name>...</function>
    FUNCTION_PATTERN = re.compile(
        r'<function(?:\s+name)?\s*[=:]\s*["\']?([^"\'>\s]+)["\']?\s*>(.*?)</function>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern to match various parameter formats:
    # - <parameter=name>value</parameter>
    # - <parameter name="name">value</parameter>
    PARAMETER_PATTERN = re.compile(
        r'<parameter(?:\s+name)?\s*[=:]\s*["\']?([^"\'>\s]+)["\']?\s*>(.*?)</parameter>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern for already-correct tool_call wrapper
    VALID_TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>(.*?)</tool_call>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern for JSON-style tool calls (some models output this)
    JSON_TOOL_CALL_PATTERN = re.compile(
        r'\{[\s\n]*"name"[\s\n]*:[\s\n]*"([^"]+)"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*(\{[^}]*\}|\[[^\]]*\]|"[^"]*")',
        re.DOTALL
    )
    
    @classmethod
    def has_malformed_tool_call(cls, content: str) -> bool:
        """Check if content has tool calls that need transformation."""
        if not content:
            return False
        
        # Already valid - no transformation needed
        if cls.VALID_TOOL_CALL_PATTERN.search(content):
            # But check if there are ALSO malformed ones outside the wrapper
            # Remove valid ones and check what's left
            remaining = cls.VALID_TOOL_CALL_PATTERN.sub('', content)
            if not cls.FUNCTION_PATTERN.search(remaining):
                return False
        
        # Has function tags without proper wrapper
        if cls.FUNCTION_PATTERN.search(content):
            return True
        
        # Has raw JSON tool calls
        if cls.JSON_TOOL_CALL_PATTERN.search(content):
            return True
        
        return False
    
    @classmethod
    def parse(cls, content: str) -> ParsedResponse:
        """
        Parse content and extract tool calls.
        
        Returns ParsedResponse with extracted tool calls and surrounding text.
        """
        if not content:
            return ParsedResponse(
                preamble="",
                tool_calls=[],
                postamble="",
                was_transformed=False
            )
        
        tool_calls = []
        
        # First, check for valid tool_call wrappers (passthrough)
        valid_matches = list(cls.VALID_TOOL_CALL_PATTERN.finditer(content))
        if valid_matches and not cls.FUNCTION_PATTERN.search(
            cls.VALID_TOOL_CALL_PATTERN.sub('', content)
        ):
            # All tool calls are properly wrapped, no transformation needed
            return ParsedResponse(
                preamble=content,  # Keep as-is
                tool_calls=[],
                postamble="",
                was_transformed=False
            )
        
        # Find all function calls (malformed)
        function_matches = list(cls.FUNCTION_PATTERN.finditer(content))
        
        if not function_matches:
            # Try JSON format
            json_matches = list(cls.JSON_TOOL_CALL_PATTERN.finditer(content))
            if json_matches:
                return cls._parse_json_tool_calls(content, json_matches)
            
            # No tool calls found
            return ParsedResponse(
                preamble=content,
                tool_calls=[],
                postamble="",
                was_transformed=False
            )
        
        # Extract preamble (text before first function)
        first_match = function_matches[0]
        preamble = content[:first_match.start()].strip()
        
        # Extract postamble (text after last function)
        last_match = function_matches[-1]
        postamble = content[last_match.end():].strip()
        
        # Parse each function call
        for match in function_matches:
            function_name = match.group(1)
            function_body = match.group(2)
            
            # Parse parameters from function body
            arguments = cls._parse_parameters(function_body)
            
            tool_calls.append(ParsedToolCall(
                function_name=function_name,
                arguments=arguments,
                raw_text=match.group(0)
            ))
        
        return ParsedResponse(
            preamble=preamble,
            tool_calls=tool_calls,
            postamble=postamble,
            was_transformed=True
        )
    
    @classmethod
    def _parse_parameters(cls, body: str) -> dict:
        """Parse parameters from function body."""
        arguments = {}
        
        for param_match in cls.PARAMETER_PATTERN.finditer(body):
            param_name = param_match.group(1)
            param_value = param_match.group(2)
            
            # Unescape XML entities
            param_value = xml_unescape(param_value)
            
            # Try to parse as JSON if it looks like JSON
            param_value = cls._maybe_parse_json(param_value)
            
            arguments[param_name] = param_value
        
        return arguments
    
    @classmethod
    def _parse_json_tool_calls(
        cls, 
        content: str, 
        matches: list
    ) -> ParsedResponse:
        """Parse JSON-style tool calls."""
        tool_calls = []
        
        first_match = matches[0]
        preamble = content[:first_match.start()].strip()
        
        last_match = matches[-1]
        postamble = content[last_match.end():].strip()
        
        for match in matches:
            function_name = match.group(1)
            arguments_str = match.group(2)
            
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {"raw": arguments_str}
            
            tool_calls.append(ParsedToolCall(
                function_name=function_name,
                arguments=arguments,
                raw_text=match.group(0)
            ))
        
        return ParsedResponse(
            preamble=preamble,
            tool_calls=tool_calls,
            postamble=postamble,
            was_transformed=True
        )
    
    @staticmethod
    def _maybe_parse_json(value: str) -> Any:
        """Try to parse value as JSON, return original if not JSON."""
        value = value.strip()
        
        if not value:
            return value
        
        # Check if it looks like JSON
        if value.startswith(('{', '[', '"')) or value in ('true', 'false', 'null'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        return value


# =============================================================================
# Response Transformer
# =============================================================================

class ResponseTransformer:
    """
    Transforms parsed tool calls into OpenAI-compatible format.
    """
    
    @staticmethod
    def generate_tool_call_id() -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"
    
    @classmethod
    def transform_response(
        cls,
        original_response: dict,
        parsed: ParsedResponse
    ) -> dict:
        """
        Transform a response with malformed tool calls into proper format.
        
        Args:
            original_response: The original API response dict
            parsed: ParsedResponse from ToolCallParser
        
        Returns:
            Transformed response dict with proper tool_calls structure
        """
        if not parsed.was_transformed or not parsed.tool_calls:
            return original_response
        
        response = json.loads(json.dumps(original_response))  # Deep copy
        
        if "choices" not in response or not response["choices"]:
            return original_response
        
        # Build tool_calls array
        tool_calls = []
        for tc in parsed.tool_calls:
            tool_calls.append({
                "id": cls.generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": tc.function_name,
                    "arguments": json.dumps(tc.arguments)
                }
            })
        
        # Update the message
        message = response["choices"][0].get("message", {})
        
        # Set content to preamble (or None if empty, per OpenAI spec)
        if parsed.preamble:
            message["content"] = parsed.preamble
        else:
            message["content"] = None
        
        # Add tool_calls
        message["tool_calls"] = tool_calls
        
        # Update finish_reason
        response["choices"][0]["finish_reason"] = "tool_calls"
        response["choices"][0]["message"] = message
        
        # Add transformation metadata (optional, for debugging)
        if "usage" not in response:
            response["usage"] = {}
        response["usage"]["_transformed"] = True
        response["usage"]["_tool_calls_extracted"] = len(tool_calls)
        
        logger.info(f"Transformed response: extracted {len(tool_calls)} tool call(s)")
        
        return response
    
    @classmethod
    def transform_streaming_content(
        cls,
        full_content: str,
        parsed: ParsedResponse,
        metadata: dict
    ) -> list[dict]:
        """
        Transform streaming content into proper SSE chunks.
        
        Returns list of chunk dicts ready for SSE serialization.
        """
        chunks = []
        chunk_id = metadata.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
        model = metadata.get("model", "unknown")
        created = int(time.time())
        
        # First chunk(s): stream the preamble content
        if parsed.preamble:
            # Split preamble into smaller chunks for realistic streaming
            chunk_size = 20
            for i in range(0, len(parsed.preamble), chunk_size):
                chunk_content = parsed.preamble[i:i + chunk_size]
                chunks.append({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None
                    }]
                })
        
        # Tool call chunks
        for idx, tc in enumerate(parsed.tool_calls):
            # Tool call start
            chunks.append({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": idx,
                            "id": cls.generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tc.function_name,
                                "arguments": ""
                            }
                        }]
                    },
                    "finish_reason": None
                }]
            })
            
            # Arguments in chunks
            args_str = json.dumps(tc.arguments)
            for i in range(0, len(args_str), 50):
                chunk_args = args_str[i:i + 50]
                chunks.append({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "function": {
                                    "arguments": chunk_args
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                })
        
        # Final chunk
        chunks.append({
            "id": chunk_id,
            "object": "chat.completion.chunk", 
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls" if parsed.tool_calls else "stop"
            }]
        })
        
        return chunks


# =============================================================================
# Request Handlers
# =============================================================================

async def handle_non_streaming_request(
    client: httpx.AsyncClient,
    body: dict
) -> dict:
    """Handle non-streaming request with transformation."""
    
    logger.info("Processing non-streaming request")
    
    response = await client.post(
        f"{config.backend_url}/v1/chat/completions",
        json=body,
        timeout=120.0
    )
    response.raise_for_status()
    result = response.json()
    
    # Extract content
    content = ""
    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")
    
    # Check if transformation is needed
    if config.transform_enabled and ToolCallParser.has_malformed_tool_call(content):
        logger.info("Detected malformed tool call, transforming...")
        
        try:
            parsed = ToolCallParser.parse(content)
            if parsed.was_transformed:
                result = ResponseTransformer.transform_response(result, parsed)
                logger.info(f"Transformation successful: {len(parsed.tool_calls)} tool call(s)")
                stats.record_transformed()
            else:
                stats.record_passthrough()
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            stats.record_failed()
            # Return original response if transformation fails
    else:
        # No transformation needed - model output was valid
        stats.record_passthrough()
        logger.debug("Passthrough: no transformation needed")
    
    return result


async def collect_stream(
    client: httpx.AsyncClient,
    body: dict
) -> tuple[str, list[dict], dict]:
    """Collect streaming response into content and chunks."""
    
    full_content = ""
    chunks = []
    metadata = {}
    
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
            
            data = line[6:]
            if data == "[DONE]":
                break
            
            try:
                chunk = json.loads(data)
                chunks.append(chunk)
                
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
                    if chunk["choices"][0].get("finish_reason"):
                        metadata["finish_reason"] = chunk["choices"][0]["finish_reason"]
                
                if "model" in chunk:
                    metadata["model"] = chunk["model"]
                if "id" in chunk:
                    metadata["id"] = chunk["id"]
                    
            except json.JSONDecodeError:
                continue
    
    return full_content, chunks, metadata


async def handle_streaming_request(
    body: dict
) -> AsyncGenerator[str, None]:
    """Handle streaming request with transformation."""
    
    logger.info("Processing streaming request")
    
    body["stream"] = True
    
    # Create client inside the generator so it stays open during iteration
    async with httpx.AsyncClient() as client:
        content, original_chunks, metadata = await collect_stream(client, body)
        
        # Log what we collected for debugging
        logger.info(f"Stream collected: {len(original_chunks)} chunks, {len(content)} chars content")
        if content:
            # Show first 200 chars to help debug
            preview = content[:200].replace('\n', '\\n')
            logger.info(f"Content preview: {preview}...")
        
        # Check if transformation needed
        has_malformed = config.transform_enabled and ToolCallParser.has_malformed_tool_call(content)
        logger.info(f"Has malformed tool call: {has_malformed}")
        
        if has_malformed:
            logger.info("Detected malformed tool call in stream, transforming...")
            
            try:
                parsed = ToolCallParser.parse(content)
                if parsed.was_transformed:
                    # Generate transformed chunks
                    transformed_chunks = ResponseTransformer.transform_streaming_content(
                        content, parsed, metadata
                    )
                    
                    for chunk in transformed_chunks:
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.005)  # Small delay for realistic streaming
                    
                    yield "data: [DONE]\n\n"
                    logger.info(f"Stream transformation successful: {len(parsed.tool_calls)} tool call(s)")
                    stats.record_transformed()
                    return
                else:
                    logger.info("Parser returned was_transformed=False, passing through")
                    stats.record_passthrough()
                    
            except Exception as e:
                logger.error(f"Stream transformation failed: {e}")
                stats.record_failed()
                # Fall through to replay original
        else:
            # No transformation needed - model output was valid (or no tool calls)
            stats.record_passthrough()
            logger.info("Stream passthrough: no malformed tool calls detected")
        
        # No transformation needed or failed - replay original chunks
        for chunk in original_chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
        logger.info(f"Stream complete. Stats: total={stats.total_requests}, passthrough={stats.passthrough_requests}, transformed={stats.transformed_requests}")


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions with transformation."""
    
    body = await request.json()
    is_streaming = body.get("stream", False)
    
    logger.info(f"Request: streaming={is_streaming}")
    
    if is_streaming:
        # For streaming, the generator manages its own client lifecycle
        return StreamingResponse(
            handle_streaming_request(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # For non-streaming, we can use context manager normally
        async with httpx.AsyncClient() as client:
            result = await handle_non_streaming_request(client, body)
            return result


@app.get("/v1/models")
async def list_models():
    """Proxy models endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{config.backend_url}/v1/models")
        return response.json()


@app.get("/health")
async def health_check():
    """Health check."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.backend_url}/health", timeout=5.0)
            backend_healthy = response.status_code == 200
    except Exception:
        backend_healthy = False
    
    return {
        "status": "healthy" if backend_healthy else "degraded",
        "backend_url": config.backend_url,
        "backend_healthy": backend_healthy,
        "transform_enabled": config.transform_enabled
    }


@app.get("/stats")
async def get_stats():
    """
    Get proxy statistics.
    
    Shows how many requests were:
    - passthrough: Model produced valid output, no transformation needed
    - transformed: Proxy fixed malformed tool calls
    - failed: Transformation was attempted but failed
    - backend_errors: Backend returned an error
    """
    return {
        "proxy_stats": stats.to_dict(),
        "raw_counts": {
            "total": stats.total_requests,
            "passthrough": stats.passthrough_requests,
            "transformed": stats.transformed_requests,
            "failed": stats.failed_transforms,
            "backend_errors": stats.backend_errors
        },
        "interpretation": {
            "passthrough": "Model output was already valid - proxy just passed it through",
            "transformed": "Model output was malformed - proxy fixed it",
            "if_all_passthrough": "Great! The model is producing valid tool calls on its own",
            "if_zero_total": "No requests recorded yet - check if requests are reaching the proxy"
        }
    }


@app.post("/stats/reset")
async def reset_stats():
    """Reset statistics counters."""
    global stats
    stats = ProxyStats()
    return {"status": "reset", "stats": stats.to_dict()}


@app.get("/proxy/test-transform")
async def test_transform(content: str = ""):
    """
    Test the transformation logic.
    
    Usage: /proxy/test-transform?content=<url-encoded-content>
    """
    if not content:
        content = """I'll help you create that file.
<function=writeFile>
<parameter=path>src/app.js</parameter>
<parameter=content>console.log("hello")</parameter>
</function>"""
    
    has_malformed = ToolCallParser.has_malformed_tool_call(content)
    parsed = ToolCallParser.parse(content)
    
    mock_response = {
        "id": "test-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    }
    
    if parsed.was_transformed:
        transformed = ResponseTransformer.transform_response(mock_response, parsed)
    else:
        transformed = mock_response
    
    return {
        "input_content": content,
        "has_malformed_tool_call": has_malformed,
        "parsed": {
            "preamble": parsed.preamble,
            "tool_calls": [
                {
                    "function_name": tc.function_name,
                    "arguments": tc.arguments
                }
                for tc in parsed.tool_calls
            ],
            "postamble": parsed.postamble,
            "was_transformed": parsed.was_transformed
        },
        "transformed_response": transformed
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str):
    """Pass through other endpoints unchanged."""
    async with httpx.AsyncClient() as client:
        url = f"{config.backend_url}/{path}"
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Response Transformer Proxy")
    parser.add_argument("--backend", "-b", default="http://localhost:8080")
    parser.add_argument("--port", "-p", type=int, default=4000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-transform", action="store_true", help="Disable transformation")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    config.backend_url = args.backend
    config.port = args.port
    config.host = args.host
    config.transform_enabled = not args.no_transform
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting LLM Response Transformer Proxy")
    logger.info(f"  Backend: {config.backend_url}")
    logger.info(f"  Listening: {config.host}:{config.port}")
    logger.info(f"  Transform: {'enabled' if config.transform_enabled else 'disabled'}")
    
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
