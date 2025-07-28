# backend/app/services/shared_services.py
import logging
import random
import math
import httpx
import time
import sqlite3
import json
from typing import Optional, List, Dict, Any, Tuple
from fastapi import HTTPException, Request

from app.schemas.vector import QueryRequest as VectorQueryRequest
from app.schemas.proxy_schemas import ApiConfig, OnlineKnowledgeBase, KnowledgeSource
from app.database import get_db_connection_for_bg

logger = logging.getLogger(__name__)

def _select_random_key(api_keys_str: str) -> str:
    """Selects a random API key from a comma-separated string."""
    keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
    if not keys:
        return ""
    return random.choice(keys)

def log_api_call(service_name: str, model_identifier: Optional[str] = None):
    """Logs an API call to the database for statistics."""
    conn = None
    try:
        conn = get_db_connection_for_bg()
        if conn:
            timestamp = int(time.time() * 1000)
            conn.execute(
                "INSERT INTO api_call_logs (service_name, model_identifier, timestamp) VALUES (?, ?, ?)",
                (service_name, model_identifier, timestamp)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log API call for service '{service_name}': {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def get_client(proxy: Optional[str] = None) -> httpx.AsyncClient:
    if proxy:
        return httpx.AsyncClient(proxy=proxy, timeout=120.0)
    return httpx.AsyncClient(timeout=120.0)

def _clean_unicode_string(s: str) -> str:
    """
    Cleans a string by removing or replacing invalid Unicode characters,
    specifically surrogate pairs that cause encoding errors.
    """
    return s.encode('utf-8', 'replace').decode('utf-8')

async def get_completion(messages: List[Dict[str, Any]], api_config: ApiConfig, tools: Optional[List[Dict]] = None, tool_choice: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    chat_assignment = api_config.assignments.chat
    if not chat_assignment:
        raise HTTPException(status_code=400, detail="Chat model is not configured in settings.")

    log_api_call("chat", chat_assignment.modelName)
    
    chat_provider = next((p for p in api_config.providers if p.id == chat_assignment.providerId), None)
    if not chat_provider:
        raise HTTPException(status_code=400, detail=f"Provider for chat model not found: {chat_assignment.providerId}")
    
    api_key = _select_random_key(chat_provider.apiKey)
    target_url = f"{chat_provider.baseUrl.strip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    forward_data = {"model": chat_assignment.modelName, "messages": messages, "stream": False}
    if tools:
        forward_data["tools"] = tools
    if tool_choice:
        forward_data["tool_choice"] = tool_choice
    if max_tokens:
        forward_data["max_tokens"] = max_tokens
    ######### Important AND don't remove, check input and output ####
    print(" =====AGENT request==== ")
    print(target_url,headers,chat_provider.proxy)
    print(forward_data)
    client = get_client(chat_provider.proxy)
    try:
        response = await client.post(target_url, headers=headers, json=forward_data)
        response.raise_for_status()
        data = response.json()
        ######### Important AND don't remove, check input and output ####

        print(" =====AGENT response==== ")
        print(data)

        # Clean the content of the response message
        if "choices" in data and data["choices"]:
            message = data["choices"][0].get("message", {})
            if "content" in message and isinstance(message["content"], str):
                message["content"] = _clean_unicode_string(message["content"])
        
        return data.get("choices", [{}])[0].get("message", {})
    finally:
        if chat_provider.proxy:
            await client.aclose()

async def get_embeddings(text: str, api_config: ApiConfig, request: Any) -> List[float]:
    embedding_assignment = api_config.assignments.embedding
    if not embedding_assignment:
        raise HTTPException(status_code=400, detail="Embedding model is not configured in settings.")

    log_api_call("embedding", embedding_assignment.modelName)

    embedding_provider = next((p for p in api_config.providers if p.id == embedding_assignment.providerId), None)
    if not embedding_provider:
        raise HTTPException(status_code=400, detail=f"Provider for embedding model not found: {embedding_assignment.providerId}")

    api_key = _select_random_key(embedding_provider.apiKey)
    embedding_payload = {
        "model": embedding_assignment.modelName,
        "input": [text],
        "provider_config": {
            "id": embedding_provider.id,
            "name": embedding_provider.name,
            "baseUrl": embedding_provider.baseUrl,
            "apiKey": api_key,
            "models": [m.model_dump() for m in embedding_provider.models],
            "proxy": embedding_provider.proxy
        }
    }
    
    base_url = api_config.execution.backendUrl if api_config.execution else str(request.base_url)
    embedding_url = f"{base_url.rstrip('/')}/api/v1/proxy/embeddings"

    async with httpx.AsyncClient() as client:
        response = await client.post(embedding_url, json=embedding_payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

async def query_knowledge_base(vector: List[float], kb_selection: str, request: Any, top_k: int, score_threshold: float, api_config: ApiConfig) -> List[Dict[str, Any]]:
    where_filter = None
    if kb_selection != "all":
        where_filter = {"file_path": {"$like": f"{kb_selection}%"}}

    query_payload = VectorQueryRequest(
        database="nexus_db",
        collection="knowledge_base",
        query_embeddings=[vector],
        n_results=top_k,
        where=where_filter,
        score_threshold=score_threshold
    )
    
    base_url = api_config.execution.backendUrl if api_config.execution else str(request.base_url)
    vector_query_url = f"{base_url.rstrip('/')}/api/v1/vector/query"

    async with httpx.AsyncClient() as client:
        response = await client.post(vector_query_url, json=query_payload.model_dump(by_alias=False))
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
                    "source_name": metadata.get("file_path", "").split("/")[-1].split("\\")[-1],
                    "content_snippet": query_results["documents"][0][i],
                    "score": score
                })
        return sources

async def query_online_kb(query: str, kb_config: OnlineKnowledgeBase, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
    logger.info(f"Querying online KB: {kb_config.name}")
    log_api_call("online_kb", kb_config.name)
    try:
        headers = {"Content-Type": "application/json"}
        if kb_config.token and kb_config.token.strip():
            headers["Authorization"] = f"Bearer {kb_config.token.strip()}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                kb_config.url,
                headers=headers,
                json={"query": query, "top_k": top_k, "score_threshold": score_threshold},
                timeout=30.0
            )
            response.raise_for_status()
            results = response.json()
            
            sources = []
            for res in results:
                sources.append({
                    "id": f"online::{kb_config.id}::{res.get('id', str(time.time()))}",
                    "file_path": f"online-kb://{kb_config.id}",
                    "source_name": res.get("source_name", kb_config.name),
                    "content_snippet": res.get("content", ""),
                    "score": res.get("score", 0.9)
                })
            return sources
    except Exception as e:
        logger.error(f"Error querying online KB '{kb_config.name}': {e}", exc_info=True)
        return []

async def tavily_search(query: str, api_config: ApiConfig) -> List[Dict[str, Any]]:
    log_api_call("search", "tavily")
    tavily_api_key = _select_random_key(api_config.keys.tavily or "")
    if not tavily_api_key:
        logger.warning("Tavily API key not configured in settings.")
        return []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "max_results": 5
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
    except Exception as e:
        logger.error(f"An unexpected error occurred during Tavily search: {e}", exc_info=True)
        return []

async def bing_search(query: str, api_config: ApiConfig) -> List[Dict[str, Any]]:
    log_api_call("search", "bing")
    bing_api_key = _select_random_key(api_config.keys.bing or "")
    if not bing_api_key:
        logger.warning("Bing Search API key not configured in settings.")
        return []

    try:
        headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
        params = {"q": query, "count": 5, "textDecorations": False, "textFormat": "Raw"}
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            web_pages = data.get("webPages", {}).get("value", [])
            # Transform Bing's response to match Tavily's structure for consistency
            return [
                {"title": r.get("name"), "url": r.get("url"), "content": r.get("snippet")}
                for r in web_pages
            ]
    except Exception as e:
        logger.error(f"An unexpected error occurred during Bing search: {e}", exc_info=True)
        return []

async def internet_search(query: str, api_config: ApiConfig) -> List[Dict[str, Any]]:
    """Performs an internet search using the default engine specified in settings."""
    engine = api_config.knowledgeBase.defaultInternetSearchEngine if api_config.knowledgeBase else "tavily"
    logger.info(f"Performing internet search for query '{query}' using engine: {engine}")
    
    if engine == "bing":
        return await bing_search(query, api_config)
    else: # Default to Tavily
        return await tavily_search(query, api_config)

async def perform_rag(query: str, kb_selection: Optional[str], api_config: Optional[ApiConfig], request: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Performs Retrieval-Augmented Generation based on the knowledge base selection."""
    if not kb_selection or kb_selection == "none" or not api_config:
        return query, []

    kb_settings = api_config.knowledgeBase
    top_k = kb_settings.topK if kb_settings else 5
    score_threshold = kb_settings.scoreThreshold if kb_settings else 0.6
    sources = []

    if kb_selection == "internet_search":
        logger.info("RAG: Internet search triggered.")
        search_results = await internet_search(query, api_config)
        
        if not search_results:
            return query, []

        # Create a KnowledgeSource for each search result
        for i, res in enumerate(search_results):
            sources.append({
                "id": f"internet_search_{i}",
                "file_path": res.get("url", ""),
                "source_name": res.get("title", "Web Search Result"),
                "content_snippet": res.get("content", ""),
                "score": 1.0 - (i * 0.01) # Assign a mock score
            })
        
        context_str = "\n\n".join([f"Source: {s['source_name']}\nContent: {s['content_snippet']}" for s in sources])
        rag_prompt = f"Use the following web search results to answer the user's question.\n\n--- WEB SEARCH RESULTS ---\n{context_str}\n--- END SEARCH RESULTS ---\n\nUser Question: {query}"
        
        return rag_prompt, sources
    
    elif kb_selection.startswith("online::"):
        kb_id = kb_selection.split("::")[1]
        online_kb_config = next((kb for kb in (api_config.onlineKbs or []) if kb.id == kb_id), None)
        if online_kb_config:
            sources = await query_online_kb(query, online_kb_config, top_k, score_threshold)
    
    else: # Local search
        logger.info(f"RAG: Local KB search triggered for selection: {kb_selection}")
        query_vector = await get_embeddings(query[-800:], api_config, request)
        sources = await query_knowledge_base(query_vector, kb_selection, request, top_k, score_threshold, api_config)

    if sources:
        logger.info(f"RAG: Retrieved {len(sources)} sources.")
        context_str = "\n\n".join([f"Source: {s['source_name']}\nContent: {s['content_snippet']}" for s in sources])
        rag_prompt = f"Use the following context to answer the user's question.\n\n--- CONTEXT ---\n{context_str}\n\n--- END CONTEXT ---\n\nUser Question: {query}"
        return rag_prompt, sources
    
    logger.info("RAG: No sources found.")
    return query, []