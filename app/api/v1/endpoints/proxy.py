# backend/app/api/v1/endpoints/proxy.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import logging
import re
from typing import Optional, List, Dict, Any
import time
import math

from pydantic import BaseModel
from app.services.vector_service import vector_service
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

http_client_no_proxy = httpx.AsyncClient(timeout=120.0)

class ApiProvider(BaseModel):
    id: str
    name: str
    baseUrl: str
    apiKey: str
    models: List[str]
    proxy: Optional[str] = None

class ModelEndpoint(BaseModel):
    providerId: str
    modelName: str

class ModelAssignments(BaseModel):
    chat: Optional[ModelEndpoint] = None
    suggestion: Optional[ModelEndpoint] = None
    vision: Optional[ModelEndpoint] = None
    imageGen: Optional[ModelEndpoint] = None
    embedding: Optional[ModelEndpoint] = None

class OtherApiKeys(BaseModel):
    tavily: Optional[str] = ""

class ApiConfig(BaseModel):
    providers: List[ApiProvider]
    assignments: ModelAssignments
    keys: OtherApiKeys

class ProviderConfig(BaseModel):
    base_url: str
    api_key: str
    proxy: Optional[str] = None

class ProxyMessage(BaseModel):
    role: str
    content: str

class ProxyChatPayload(BaseModel):
    model: str
    messages: List[ProxyMessage]
    stream: bool = False
    provider_config: Optional[ProviderConfig] = None # Make optional for translation
    knowledge_base_selection: Optional[str] = None
    api_config: Optional[ApiConfig] = None # Make optional

class ProxyEmbeddingPayload(BaseModel):
    model: str
    input: List[str]
    provider_config: ProviderConfig

class KnowledgeSource(BaseModel):
    id: str
    file_path: str
    source_name: str
    content_snippet: str
    score: float

class StreamChoice(BaseModel):
    delta: Dict[str, Any]
    
class ProxyStreamChunk(BaseModel):
    choices: List[StreamChoice]
    sources: Optional[List[KnowledgeSource]] = None

class TranslateRequest(BaseModel):
    content: str
    from_lang: Optional[str] = "auto"
    to_lang: Optional[str] = "en"

class SearchRequest(BaseModel):
    query: str
    api_config: ApiConfig

def get_client(proxy: Optional[str] = None) -> httpx.AsyncClient:
    if proxy:
        proxies = {"http://": proxy, "https://": proxy}
        return httpx.AsyncClient(proxies=proxies, timeout=120.0)
    return http_client_no_proxy

async def stream_request(method: str, url: str, headers: dict, json_data: dict, proxy: Optional[str], sources: Optional[List[Dict[str, Any]]]):
    client = get_client(proxy)
    logger.info(f"Streaming to {url} with proxy: {proxy or 'None'}")
    
    try:
        if sources:
            chunk_with_sources = ProxyStreamChunk(choices=[], sources=[KnowledgeSource(**s) for s in sources])
            yield f"data: {chunk_with_sources.model_dump_json()}\n\n"

        async with client.stream(method, url, headers=headers, json=json_data) as response:
            logger.info(f"Target API response status: {response.status_code}")
            if response.status_code >= 400:
                error_body = await response.aread()
                logger.error(f"Target API returned error {response.status_code}: {error_body.decode()}")
                error_payload = {"error": {"message": f"Target API Error ({response.status_code}): {error_body.decode()}", "type": "api_error"}}
                yield f"data: {json.dumps(error_payload)}\n\n"
                return

            async for chunk in response.aiter_text():
                if chunk:
                    yield chunk
    except httpx.ConnectError as e:
        logger.error(f"Connection error while streaming from {url}: {e}", exc_info=True)
        error_payload = {"error": {"message": f"Failed to connect to the target API: {e}", "type": "connection_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        logger.error(f"Generic error during stream from {url}: {e}", exc_info=True)
        error_payload = {"error": {"message": f"An unexpected error occurred during streaming: {e}", "type": "streaming_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n"
    finally:
        if proxy:
            await client.aclose()

async def get_embeddings(text: str, api_config: ApiConfig, request: Request) -> List[float]:
    embedding_assignment = api_config.assignments.embedding
    if not embedding_assignment:
        raise HTTPException(status_code=400, detail="Embedding model is not configured in settings.")

    embedding_provider = next((p for p in api_config.providers if p.id == embedding_assignment.providerId), None)
    if not embedding_provider:
        raise HTTPException(status_code=400, detail=f"Provider for embedding model not found: {embedding_assignment.providerId}")

    embedding_payload = {
        "model": embedding_assignment.modelName,
        "input": [text],
        "provider_config": {
            "base_url": embedding_provider.baseUrl,
            "api_key": embedding_provider.apiKey,
            "proxy": embedding_provider.proxy
        }
    }
    embedding_url = str(request.url_for('proxy_embeddings'))
    async with httpx.AsyncClient() as client:
        response = await client.post(embedding_url, json=embedding_payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

async def query_knowledge_base(vector: List[float], kb_selection: str, request: Request) -> List[Dict[str, Any]]:
    ids_to_search = None
    if kb_selection != "all":
        ids_to_search = vector_service.get_ids_by_path_prefix("knowledge_base", kb_selection)
        if not ids_to_search:
            return []

    query_payload = {
        "database": "nexus_db",
        "collection": "knowledge_base",
        "query_embeddings": [vector],
        "n_results": 3,
        "ids": ids_to_search
    }
    vector_query_url = str(request.url_for('query_vectors'))
    async with httpx.AsyncClient() as client:
        response = await client.post(vector_query_url, json=query_payload)
        response.raise_for_status()
        query_results = response.json()
        
        sources = []
        if query_results.get("ids") and query_results["ids"][0]:
            for i, doc_id in enumerate(query_results["ids"][0]):
                metadata = query_results["metadatas"][0][i]
                distance = query_results["distances"][0][i]
                score = math.exp(-distance)
                
                sources.append({
                    "id": doc_id,
                    "file_path": metadata.get("file_path", ""),
                    "source_name": metadata.get("file_path", "").split("/")[-1],
                    "content_snippet": query_results["documents"][0][i],
                    "score": score
                })
        return sources

async def tavily_search(query: str, api_config: ApiConfig) -> str:
    tavily_api_key = api_config.keys.tavily
    if not tavily_api_key:
        return "Tavily API key not configured in settings."
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 3
            }
        )
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            results = "\n".join([f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}" for r in data.get("results", [])])
            return f"Search Answer: {answer}\n\nResults:\n{results}"
        else:
            return f"Tavily search failed with status {response.status_code}: {response.text}"

async def perform_translation(text: str) -> str:
    from_lang = "auto"
    to_lang = "en"
    
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    if len(pattern.findall(text)) > len(text) / 4:
        from_lang, to_lang = "zh", "en"
    else:
        from_lang, to_lang = "en", "zh"

    url = f'https://translate.googleapis.com/translate_a/single?client=gtx&sl={from_lang}&tl={to_lang}&dt=t&q={httpx.utils.quote(text)}'
    
    proxy_to_use = "socks5://127.0.0.1:1089"
    logger.info(f"Translation request to Google. Using proxy: {proxy_to_use}")
    
    try:
        async with get_client(proxy_to_use) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            translated_text = "".join([sentence.get("trans", "") for sentence in data[0]])
            return translated_text
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Translation service failed.")

@router.post("/chat/completions")
async def proxy_chat_completions(payload: ProxyChatPayload, request: Request):
    logger.info(f"Received chat completion request for model: {payload.model}")

    if payload.model == "translation":
        if not payload.messages:
            raise HTTPException(status_code=400, detail="No messages provided for translation.")
        
        text_to_translate = payload.messages[-1].content
        translated_text = await perform_translation(text_to_translate)
        
        response_data = {
            "id": f"trans-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "translation-service",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": translated_text}, "finish_reason": "stop"}],
        }
        return JSONResponse(content=response_data)

    if not payload.provider_config or not payload.api_config:
        raise HTTPException(status_code=400, detail="Provider and API config are required for LLM calls.")

    messages = [msg.model_dump() for msg in payload.messages]
    retrieved_sources = None
    user_question = messages[-1]["content"]
    kb_selection = payload.knowledge_base_selection

    if kb_selection == "internet_search":
        logger.info("Internet search triggered by knowledge selection.")
        search_results = await tavily_search(user_question, payload.api_config)
        context_str = f"--- WEB SEARCH RESULTS ---\n{search_results}\n--- END SEARCH RESULTS ---"
        rag_prompt = f"Use the following web search results to answer the user's question.\n\n{context_str}\n\nUser Question: {user_question}"
        messages[-1]["content"] = rag_prompt
    elif kb_selection and kb_selection != "none":
        logger.info(f"RAG triggered for local KB selection: {kb_selection}")
        try:
            query_vector = await get_embeddings(user_question[-800:], payload.api_config, request)
            retrieved_sources = await query_knowledge_base(query_vector, kb_selection, request)
            if retrieved_sources:
                logger.info(f"Retrieved {len(retrieved_sources)} sources from local KB.")
                context_str = "\n\n".join([f"Source: {s['file_path']}\nContent: {s['content_snippet']}" for s in retrieved_sources])
                rag_prompt = f"Use the following context to answer the user's question.\n\n--- CONTEXT ---\n{context_str}\n\n--- END CONTEXT ---\n\nUser Question: {user_question}"
                messages[-1]["content"] = rag_prompt
        except Exception as e:
            logger.error(f"Error during local RAG processing: {e}", exc_info=True)

    try:
        provider = payload.provider_config
        target_url = f"{provider.base_url.strip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json", "Accept": "text/event-stream" if payload.stream else "application/json"}
        forward_data = {"model": payload.model, "messages": messages, "stream": payload.stream}
        
        logger.info(f"Forwarding request to: {target_url}")

        if payload.stream:
            return StreamingResponse(stream_request("POST", target_url, headers=headers, json_data=forward_data, proxy=provider.proxy, sources=retrieved_sources), media_type="text/event-stream")
        else:
            client = get_client(provider.proxy)
            try:
                response = await client.post(target_url, headers=headers, json=forward_data)
                response.raise_for_status()
                return JSONResponse(content=response.json())
            finally:
                if provider.proxy:
                    await client.aclose()

    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"HTTP Status Error from target API: {e.response.status_code} - {error_content}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from target API: {error_content}")
    except Exception as e:
        logger.error(f"Unhandled exception in proxy_chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.post("/embeddings")
async def proxy_embeddings(payload: ProxyEmbeddingPayload):
    logger.info(f"Received embedding request for model: {payload.model}")
    try:
        provider = payload.provider_config
        target_url = f"{provider.base_url.strip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}
        forward_data = {"model": payload.model, "input": payload.input}
        
        logger.info(f"Forwarding clean embedding request to: {target_url}")

        client = get_client(provider.proxy)
        try:
            response = await client.post(target_url, headers=headers, json=forward_data)
            response.raise_for_status()
            return JSONResponse(content=response.json())
        finally:
            if provider.proxy:
                await client.aclose()

    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"HTTP Status Error from embedding API: {e.response.status_code} - {error_content}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from target API: {error_content}")
    except Exception as e:
        logger.error(f"Unhandled exception in proxy_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.post("/search")
async def search(req: SearchRequest):
    try:
        result = await tavily_search(req.query, req.api_config)
        return {"search_result": result}
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search service failed.")