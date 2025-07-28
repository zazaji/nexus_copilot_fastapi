# backend/app/api/v1/endpoints/proxy.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import logging
from typing import Optional, List, Dict, Any

from app.services import shared_services
from app.schemas.proxy_schemas import ProxyChatPayload, ProxyEmbeddingPayload, SearchRequest, ModelInfo, ApiProvider

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _format_messages_for_model(messages: List[Dict[str, Any]], model_info: Optional[ModelInfo]) -> List[Dict[str, Any]]:
    """
    Dynamically formats the 'content' of each message in the history
    based on the target model's capabilities.
    """
    if not model_info: # If model info is missing, assume text-only for safety
        supports_vision = False
    else:
        supports_vision = 'vision' in model_info.capabilities

    formatted_messages = []
    for msg in messages:
        content = msg.get("content")
        
        if supports_vision:
            # Model supports vision: ensure content is an array.
            if isinstance(content, str):
                formatted_content = [{"type": "text", "text": content}]
            else:
                # Already in array format or is None/empty
                formatted_content = content
        else:
            # Model is text-only: ensure content is a string.
            if isinstance(content, list):
                # Extract text from a list of content parts
                text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                formatted_content = "\n".join(text_parts)
            else:
                # Already a string or is None/empty
                formatted_content = content
        
        formatted_messages.append({"role": msg["role"], "content": formatted_content})
        
    return formatted_messages


async def stream_request(method: str, url: str, headers: dict, json_data: dict, proxy: Optional[str]):
    client = shared_services.get_client(proxy)
    logger.info(f"Streaming to {url} with proxy: {proxy or 'None'}")

    try:
        async with client.stream(method, url, headers=headers, json=json_data, timeout=120.0) as response:
            logger.info(f"Target API response status: {response.status_code}")
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.HTTPStatusError as e:
        error_body_bytes = await e.response.aread()
        error_body = error_body_bytes.decode('utf-8')
        logger.error(f"Target API returned error {e.response.status_code}: {error_body}")
        error_payload = {"error": {"message": f"Target API Error ({e.response.status_code}): {error_body}", "type": "api_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n".encode('utf-8')
    except Exception as e:
        logger.error(f"Generic error during stream from {url}: {e}", exc_info=True)
        error_payload = {"error": {"message": f"An unexpected error occurred during streaming: {e}", "type": "streaming_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n".encode('utf-8')
    finally:
        if proxy:
            await client.aclose()

def _extract_text_from_content(content: Any) -> str:
    """Extracts plain text from a multimodal content structure."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return ""

# --- Main Endpoints ---

@router.post("/chat/completions")
async def proxy_chat_completions(payload: ProxyChatPayload, request: Request):
    logger.info(f"Received chat completion request for model: {payload.model} (stream: {payload.stream})")
    shared_services.log_api_call("chat", payload.model)
    
    messages = [msg.model_dump() for msg in payload.messages if msg.content]
    if not messages:
        raise HTTPException(status_code=400, detail="No valid messages with content provided.")

    last_user_message_content = messages[-1]["content"]
    user_question_for_rag = _extract_text_from_content(last_user_message_content)
    kb_selection = payload.knowledge_base_selection
    
    augmented_query, sources = await shared_services.perform_rag(user_question_for_rag, kb_selection, payload.api_config, request)
    
    # If RAG was performed, update the text part of the last message
    if sources and isinstance(last_user_message_content, list):
        for part in last_user_message_content:
            if part.get("type") == "text":
                part["text"] = augmented_query
                break
    elif sources: # content was a simple string
        messages[-1]["content"] = augmented_query

    provider = payload.provider_config
    target_url = f"{provider.baseUrl.strip('/')}/chat/completions"
    api_key: str = shared_services._select_random_key(provider.apiKey)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if payload.stream:
        headers["Accept"] = "text/event-stream"

    # Get model info to determine its capabilities and max_tokens
    model_info = next((m for m in provider.models if m.name == payload.model), None)

    # Format message history based on model capabilities
    formatted_messages = _format_messages_for_model(messages, model_info)

    forward_data = {
        "model": payload.model,
        "messages": formatted_messages,
        "stream": payload.stream
    }
    
    # Add max_tokens if it's configured for the model
    if model_info and model_info.max_tokens:
        forward_data["max_tokens"] = model_info.max_tokens

    #########important don't remove#########
    print(target_url,headers,provider.proxy)
    print(forward_data)
    #######################################
    async def stream_generator():
        if sources:
            sources_json = json.dumps([s for s in sources])
            yield f"event: sources\ndata: {sources_json}\n\n"
        async for chunk in stream_request("POST", target_url, headers=headers, json_data=forward_data, proxy=provider.proxy):
            yield chunk

    if payload.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        client = shared_services.get_client(provider.proxy)
        try:
            response = await client.post(target_url, headers=headers, json=forward_data, timeout=120.0)
            response.raise_for_status()
            response_data = response.json()
            response_data['sources'] = sources
            return JSONResponse(content=response_data)
        except httpx.HTTPStatusError as e:
            logger.error(f"Target API returned error {e.response.status_code}: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Error during non-stream request from {target_url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if provider.proxy:
                await client.aclose()

@router.post("/embeddings")
async def proxy_embeddings(payload: ProxyEmbeddingPayload):
    logger.info(f"Received embedding request for model: {payload.model}")
    try:
        provider = payload.provider_config
        target_url = f"{provider.baseUrl.strip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {provider.apiKey}", "Content-Type": "application/json"}
        forward_data = {"model": payload.model, "input": payload.input}

        client = shared_services.get_client(provider.proxy)
        try:
            response = await client.post(target_url, headers=headers, json=forward_data)
            response.raise_for_status()
            return JSONResponse(content=response.json())
        finally:
            if provider.proxy:
                await client.aclose()
    except Exception as e:
        logger.error(f"Unhandled exception in proxy_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.post("/search")
async def search(req: SearchRequest):
    try:
        result = await shared_services.tavily_search(req.query, req.api_config)
        return {"search_result": result}
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search service failed.")