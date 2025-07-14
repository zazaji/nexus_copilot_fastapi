# backend/app/services/vector_service.py
import chromadb
from chromadb.api.client import Client
from ..core.config import settings
from ..schemas.vector import AddRequest, QueryRequest, DeleteRequest, QueryResponse, GetAllResponse
from typing import Optional, Any, Dict, List
import os

class VectorService:
    _client: Client
    _persist_path: str

    def __init__(self):
        self._persist_path = settings.CHROMA_PERSIST_PATH
        os.makedirs(self._persist_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_path)
        
        print("="*50)
        print(f"ChromaDB service initialized in persistent mode.")
        print(f"Data path: {os.path.abspath(self._persist_path)}")
        print(f"ChromaDB version: {chromadb.__version__}")
        print("="*50)

    def get_storage_size(self) -> int:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self._persist_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def ensure_collection(self, db_name: str, collection_name: str):
        self._client.get_or_create_collection(name=collection_name)
        print(f"Ensured collection '{collection_name}' exists.")

    def add(self, req: AddRequest):
        collection = self._client.get_collection(name=req.collection)
        collection.add(
            ids=req.ids,
            embeddings=req.embeddings,
            documents=req.documents,
            metadatas=req.metadatas
        )
        print(f"Added {len(req.ids)} documents to collection '{req.collection}'.")

    def get_ids_by_path_prefix(self, collection_name: str, path_prefix: str) -> List[str]:
        """
        Gets all document IDs where the 'file_path' metadata starts with the given prefix.
        This is a robust alternative to using the '$like' operator.
        """
        collection = self._client.get_collection(name=collection_name)
        # Get all documents with their metadata
        all_docs = collection.get(include=["metadatas"])
        
        filtered_ids = []
        for i, metadata in enumerate(all_docs['metadatas']):
            if metadata and 'file_path' in metadata:
                if str(metadata['file_path']).startswith(path_prefix):
                    filtered_ids.append(all_docs['ids'][i])
        
        return filtered_ids

    def query(self, req: QueryRequest) -> QueryResponse:
        collection = self._client.get_collection(name=req.collection)
        print(f"Querying collection '{req.collection}' with {len(req.query_embeddings)} embeddings, filter: {req.where}, ids: {req.ids is not None}")
        
        query_params = {
            "query_embeddings": req.query_embeddings,
            "n_results": req.n_results,
        }
        if req.where is not None:
            query_params["where"] = req.where
        if req.ids is not None:
            # If an empty list of IDs is provided, it means no documents matched the pre-filter.
            # ChromaDB query with empty `ids` list might error or return all results, so we handle it.
            if not req.ids:
                return QueryResponse(ids=[[]], documents=[[]], metadatas=[[]], distances=[[]])
            query_params["ids"] = req.ids
            
        results = collection.query(**query_params)
        
        return QueryResponse(
            ids=results.get('ids', []),
            documents=results.get('documents', []),
            metadatas=results.get('metadatas', []),
            distances=results.get('distances', [])
        )

    def delete(self, req: DeleteRequest):
        collection = self._client.get_collection(name=req.collection)
        collection.delete(where=req.where)
        print(f"Deleted documents from '{req.collection}' where metadata matches {req.where}.")

    def update_metadata(self, collection_name: str, where: Dict[str, Any], new_metadata: Dict[str, Any]):
        collection = self._client.get_collection(name=collection_name)
        docs = collection.get(where=where, include=["metadatas"])
        if not docs['ids']:
            print(f"No documents found to update for filter: {where}")
            return

        updated_metadatas = []
        for metadata in docs['metadatas']:
            new_meta = metadata.copy()
            new_meta.update(new_metadata)
            updated_metadatas.append(new_meta)
        
        collection.update(ids=docs['ids'], metadatas=updated_metadatas)
        print(f"Updated metadata for {len(docs['ids'])} documents.")

    def clear_collection(self, collection_name: str):
        try:
            self._client.delete_collection(name=collection_name)
            print(f"Deleted collection '{collection_name}'.")
        except ValueError:
            print(f"Collection '{collection_name}' did not exist, nothing to delete.")
        self._client.create_collection(name=collection_name)
        print(f"Recreated collection '{collection_name}'.")

    def get_all(self, collection_name: str) -> GetAllResponse:
        collection = self._client.get_collection(name=collection_name)
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        
        return GetAllResponse(
            ids=results.get('ids', []),
            documents=results.get('documents', []),
            metadatas=results.get('metadatas', []),
            embeddings=results.get('embeddings', [])
        )

    def count(self, collection_name: str) -> int:
        collection = self._client.get_collection(name=collection_name)
        return collection.count()

vector_service = VectorService()