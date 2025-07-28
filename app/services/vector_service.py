# backend/app/services/vector_service.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.client import Client
from ..core.config import settings
from ..schemas.vector import AddRequest, QueryRequest, DeleteRequest, QueryResponse, GetAllResponse
from typing import Optional, Any, Dict, List
import os
import logging
import math

class VectorService:
    _client: Optional[Client] = None
    _persist_path: str

    def __init__(self):
        self._persist_path = settings.CHROMA_PERSIST_PATH
        os.makedirs(self._persist_path, exist_ok=True)
        try:
            # Initialize ChromaDB with telemetry disabled
            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=self._persist_path
            )
            self._client = chromadb.Client(chroma_settings)
            
            # Try to get a collection to test the connection
            self._client.get_or_create_collection(name="connection_test")
            self._client.delete_collection(name="connection_test")
            logging.info("="*50)
            logging.info(f"ChromaDB service initialized and connected successfully.")
            logging.info(f"Data path: {os.path.abspath(self._persist_path)}")
            logging.info(f"Anonymized telemetry: DISABLED")
            logging.info(f"ChromaDB version: {chromadb.__version__}")
            logging.info("="*50)
        except Exception as e:
            self._client = None
            logging.error("="*50)
            logging.error("!!! CHROMA DB CONNECTION FAILED !!!")
            logging.error(f"Error: {e}")
            logging.error("The vector service will be disabled. Please ensure ChromaDB is running.")
            logging.error("="*50)

    def _get_collection(self, collection_name: str):
        if not self._client:
            raise Exception("ChromaDB service is not available.")
        return self._client.get_or_create_collection(name=collection_name)

    def get_storage_size(self) -> int:
        if not self._client:
            return 0
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self._persist_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def ensure_collection(self, db_name: str, collection_name: str):
        self._get_collection(collection_name)
        logging.info(f"Ensured collection '{collection_name}' exists.")

    def add(self, req: AddRequest):
        collection = self._get_collection(req.collection)
        collection.add(
            ids=req.ids,
            embeddings=req.embeddings,
            documents=req.documents,
            metadatas=req.metadatas
        )
        logging.info(f"Added {len(req.ids)} documents to collection '{req.collection}'.")

    def query(self, req: QueryRequest) -> QueryResponse:
        collection = self._get_collection(req.collection)
        
        path_prefix_filter = None
        final_where = req.where
        n_results_to_fetch = req.n_results

        if req.where and "file_path" in req.where and isinstance(req.where["file_path"], dict) and "$like" in req.where["file_path"]:
            path_prefix_filter = req.where["file_path"]["$like"].replace("%", "")
            final_where = None
            n_results_to_fetch = min(req.n_results * 10, 100)
            logging.info(f"Performing post-filtering for path prefix: '{path_prefix_filter}'. Fetching {n_results_to_fetch} candidates.")
        
        logging.info(f"Querying collection '{req.collection}' with {len(req.query_embeddings)} embeddings, filter: {final_where}, n_results: {n_results_to_fetch}")
        
        query_params = {
            "query_embeddings": req.query_embeddings,
            "n_results": n_results_to_fetch,
        }
        if final_where is not None:
            query_params["where"] = final_where
        if req.ids is not None:
            if not req.ids:
                return QueryResponse(ids=[[]], documents=[[]], metadatas=[[]], distances=[[]])
            query_params["ids"] = req.ids
            
        results = collection.query(**query_params)

        # --- Start of Filtering Logic ---
        filtered_results = {
            'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]
        }

        if results['ids'] and results['ids'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                # Path prefix filter
                if path_prefix_filter:
                    if not (metadata and metadata.get('file_path', '').startswith(path_prefix_filter)):
                        continue
                
                # Score threshold filter
                if req.score_threshold is not None:
                    distance = results['distances'][0][i]
                    score = math.exp(-distance) # Convert distance to similarity score
                    if score < req.score_threshold:
                        continue

                # If all filters pass, add the result
                filtered_results['ids'][0].append(results['ids'][0][i])
                filtered_results['documents'][0].append(results['documents'][0][i])
                filtered_results['metadatas'][0].append(metadata)
                filtered_results['distances'][0].append(results['distances'][0][i])

        # Truncate to the original requested n_results (top_k) AFTER filtering
        for key in filtered_results:
            filtered_results[key][0] = filtered_results[key][0][:req.n_results]
        # --- End of Filtering Logic ---

        return QueryResponse(
            ids=filtered_results.get('ids', []),
            documents=filtered_results.get('documents', []),
            metadatas=filtered_results.get('metadatas', []),
            distances=filtered_results.get('distances', [])
        )

    def delete(self, req: DeleteRequest):
        collection = self._get_collection(req.collection)
        collection.delete(where=req.where)
        logging.info(f"Deleted documents from '{req.collection}' where metadata matches {req.where}.")

    def count(self, collection_name: str) -> int:
        try:
            collection = self._get_collection(collection_name)
            return collection.count()
        except Exception as e:
            logging.error(f"Failed to count vectors in '{collection_name}': {e}")
            return 0

    def clear_collection(self, collection_name: str):
        if not self._client:
            raise Exception("ChromaDB service is not available.")
        try:
            self._client.delete_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' cleared successfully.")
        except Exception as e:
            logging.error(f"Failed to clear collection '{collection_name}': {e}")
            raise e
            
    def get_all(self, collection_name: str) -> GetAllResponse:
        collection = self._get_collection(collection_name)
        results = collection.get(include=["metadatas", "documents", "embeddings"])
        return GetAllResponse(
            ids=results.get('ids', []),
            documents=results.get('documents', []),
            metadatas=results.get('metadatas', []),
            embeddings=results.get('embeddings', [])
        )

vector_service = VectorService()