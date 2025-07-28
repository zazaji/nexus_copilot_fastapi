# backend/app/knowledge_base/indexer.py
from app.services import parser_service
from ..services import vector_client
from ..services.proxy_types import EmbeddingResponse, ProxyEmbeddingPayload, ProxyProviderConfig
from ..database.models import ApiProvider
from ..state import AppState
from ..database import queries as db_queries
from ..error import AppError
from tauri import AppHandle, Manager
from uuid import uuid4
from walkdir import WalkDir
from typing import List
import logging
import os

BATCH_SIZE = 32

async def get_embeddings_from_proxy(state: AppState, provider_config: ApiProvider, model_name: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    
    settings = db_queries.get_settings(state.db.lock().unwrap())
    backend_url = settings.execution.backend_url
    url = f"{backend_url}/api/v1/proxy/embeddings"
    
    proxy_config = ProxyProviderConfig.from_orm(provider_config)
    payload = ProxyEmbeddingPayload(model=model_name, input=texts, provider_config=proxy_config)

    async with state.http_client.post(url, json=payload.dict()) as response:
        response.raise_for_status()
        response_data = await response.json()
        embedding_response = EmbeddingResponse(**response_data)
        embedding_response.data.sort(key=lambda d: d.index)
        return [d.embedding for d in embedding_response.data]

async def reindex_file(state: AppState, path_str: str):
    logging.info(f"Re-indexing single file: {path_str}")
    path = os.path.normpath(path_str)
    
    await delete_documents_for_path(state, path)
    await process_file(state, path)
    
    logging.info(f"Finished re-indexing file: {path}")

async def index_directory(app: AppHandle, state: AppState, path: str):
    logging.info(f"Starting indexing for path: {path}")
    
    settings = db_queries.get_settings(state.db.lock().unwrap())
    backend_url = settings.execution.backend_url
    await vector_client.ensure_collection(state, backend_url)

    root_path = os.path.normpath(path)
    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(root_path) for file in files]

    total_files = len(files_to_process)
    for i, file_path in enumerate(files_to_process):
        progress = ((i + 1) / total_files) * 100.0
        app.emit_all(
            "indexing-progress",
            {"file": file_path, "progress": progress}
        )
        try:
            await process_file(state, file_path)
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")

    logging.info(f"Finished indexing for path: {path}")

async def process_file(state: AppState, path: str):
    try:
        content = parser_service.parse_file(path)
    except AppError as e:
        logging.warning(f"Skipping file {path} due to parsing error: {e}")
        return

    chunks = [s for s in content.split("\n\n") if s.strip()]
    if not chunks:
        return

    settings = db_queries.get_settings(state.db.lock().unwrap())
    embedding_endpoint = settings.api_config.assignments.embedding
    if not embedding_endpoint:
        raise AppError("Config", "Embedding model not assigned")
    
    provider = next((p for p in settings.api_config.providers if p.id == embedding_endpoint.provider_id), None)
    if not provider:
        raise AppError("Config", "Embedding provider not found")

    model_name = embedding_endpoint.model_name
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        embeddings = await get_embeddings_from_proxy(state, provider, model_name, batch)
        if not embeddings:
            continue

        ids = [str(uuid4()) for _ in batch]
        metadatas = [{"file_path": path} for _ in batch]
        
        payload = vector_client.AddPayload(
            base=vector_client.VectorBase(database="nexus_db", collection="knowledge_base"),
            ids=ids,
            embeddings=embeddings,
            documents=batch,
            metadatas=metadatas,
        )
        await vector_client.add(state, settings.execution.backend_url, payload)

async def delete_documents_for_path(state: AppState, path: str):
    logging.info(f"Deleting documents for path: {path}")
    settings = db_queries.get_settings(state.db.lock().unwrap())
    backend_url = settings.execution.backend_url

    payload = vector_client.DeletePayload(
        base=vector_client.VectorBase(database="nexus_db", collection="knowledge_base"),
        where_metadata={"file_path": path},
    )
    await vector_client.delete(state, backend_url, payload)
    logging.info(f"Successfully deleted documents for path: {path}")

async def update_path_in_vector_db(state: AppState, old_path: str, new_path: str):
    logging.info(f"Updating file path in vector DB from {old_path} to {new_path}")
    settings = db_queries.get_settings(state.db.lock().unwrap())
    backend_url = settings.execution.backend_url
    
    payload = vector_client.UpdateMetadataPayload(
        base=vector_client.VectorBase(database="nexus_db", collection="knowledge_base"),
        where_metadata={"file_path": old_path},
        new_metadata={"file_path": new_path},
    )
    await vector_client.update_metadata(state, backend_url, payload)

async def clear_collection(state: AppState):
    logging.info("Clearing entire knowledge base collection.")
    settings = db_queries.get_settings(state.db.lock().unwrap())
    backend_url = settings.execution.backend_url
    await vector_client.clear_collection(state, backend_url)