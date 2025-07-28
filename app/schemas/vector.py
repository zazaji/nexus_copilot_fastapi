# backend/app/schemas/vector.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class VectorBase(BaseModel):
    database: str = "nexus_db"
    collection: str = "knowledge_base"

class EnsureCollectionRequest(VectorBase):
    pass

class AddRequest(VectorBase):
    ids: List[str]
    embeddings: List[List[float]]
    documents: List[str]
    metadatas: List[Dict[str, Any]]

class QueryRequest(BaseModel):
    database: str = "nexus_db"
    collection: str = "knowledge_base"
    query_embeddings: List[List[float]]
    n_results: int = 5
    where: Optional[Dict[str, Any]] = Field(None)
    score_threshold: Optional[float] = Field(None)
    ids: Optional[List[str]] = None

class DeleteRequest(VectorBase):
    where: Dict[str, Any] = Field(..., alias="where")

class QueryResponse(BaseModel):
    ids: List[List[str]]
    documents: List[Optional[List[Optional[str]]]]
    metadatas: List[Optional[List[Optional[Dict[str, Any]]]]]
    distances: List[Optional[List[float]]]

class GetAllResponse(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    embeddings: List[List[float]]

class CountResponse(BaseModel):
    count: int

class SizeResponse(BaseModel):
    size_bytes: int