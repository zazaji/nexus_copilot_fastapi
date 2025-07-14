# backend/app/api/v1/endpoints/dev.py
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator

router = APIRouter()

# --- Mock Chat Completions Endpoint ---

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False

async def generate_and_stream_response() -> AsyncGenerator[str, None]:
    """
    An async generator that yields formatted server-sent events.
    """
    text = "This is a local streaming test from the FastAPI mock server using yield. "
    chunks = text.split(" ")
    
    for i, chunk in enumerate(chunks):
        response_chunk = {
            "id": "chatcmpl-mock-yield-123",
            "object": "chat.completion.chunk",
            "created": 1694268190,
            "model": "local-test-model-yield",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"{chunk} "},
                    "finish_reason": None,
                }
            ],
        }
        # Format as a Server-Sent Event (SSE)
        yield f"data: {json.dumps(response_chunk)}\n\n"
        await asyncio.sleep(0.1)  # Simulate network delay

    # Signal the end of the stream with the [DONE] message
    yield "data: [DONE]\n\n"

@router.post("/chat/completions", summary="Mock Chat Completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    A mock endpoint that simulates OpenAI's chat completions API with streaming.
    This version uses an async generator directly returned by the endpoint.
    """
    print(f"Received mock chat completion request for model: {request.model}")
    
    if request.stream:
        # FastAPI automatically handles the async generator and creates a StreamingResponse
        return StreamingResponse(generate_and_stream_response(), media_type="text/event-stream")
    else:
        # Non-streaming response remains the same
        return {
            "id": "chatcmpl-mock-nonstream",
            "object": "chat.completion",
            "created": 1694268190,
            "model": "local-test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a non-streaming response from the FastAPI mock server.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }