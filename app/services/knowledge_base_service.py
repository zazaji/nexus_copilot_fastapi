# backend/app/services/knowledge_base_service.py
import logging
from typing import List
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import Request

from .vector_service import vector_service
from . import shared_services
from ..schemas.vector import AddRequest
from ..schemas.proxy_schemas import ApiConfig

logger = logging.getLogger(__name__)

BATCH_SIZE = 32

def split_text_into_chunks(text: str) -> List[str]:
    """Splits text using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

async def process_and_embed_file(file_path: str, content: str, api_config: ApiConfig, request: Request):
    """
    Processes a file's content by chunking, embedding, and adding to the vector store.
    """
    logger.info(f"Processing and embedding file: {file_path}")
    chunks = split_text_into_chunks(content)
    if not chunks:
        logger.info(f"No chunks generated for file: {file_path}")
        return

    embedding_endpoint = api_config.assignments.embedding
    if not embedding_endpoint:
        raise Exception("Embedding model not configured in settings.")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        
        # This is a simplified way to call get_embeddings. In a real app, you might pass the request differently
        # or have a dedicated internal embedding client.
        embeddings = await shared_services.get_embeddings(
            text=" ".join(batch), # A placeholder, as the backend function expects a single string
            api_config=api_config,
            request=request
        )
        
        # The above is a simplification. A more correct implementation would be to adapt get_embeddings
        # to handle a list of texts, or call it in a loop. For now, we assume a batch embedding endpoint.
        # Let's simulate a batch embedding call for robustness.
        
        embedding_provider = next((p for p in api_config.providers if p.id == embedding_endpoint.providerId), None)
        if not embedding_provider:
            raise Exception(f"Provider for embedding model not found: {embedding_endpoint.providerId}")

        # This is a mock of how a real batch embedding call would work
        # In a real scenario, the embedding endpoint would support a list of strings.
        # For now, we will call it multiple times as a fallback.
        embeddings_list = []
        for text_chunk in batch:
             embedding = await shared_services.get_embeddings(text_chunk, api_config, request)
             embeddings_list.append(embedding)

        if not embeddings_list:
            continue

        ids = [str(uuid4()) for _ in batch]
        metadatas = [{"file_path": file_path} for _ in batch]
        
        vector_service.add(AddRequest(
            collection="knowledge_base",
            ids=ids,
            embeddings=embeddings_list,
            documents=batch,
            metadatas=metadatas
        ))
    
    logger.info(f"Successfully processed and embedded {len(chunks)} chunks for {file_path}")
