# backend/app/api/v1/endpoints/vector.py
from fastapi import APIRouter, HTTPException, Body
from app.schemas import vector as vector_schemas
from app.services.vector_service import vector_service
from typing import Dict, Any

router = APIRouter()

class UpdateMetadataRequest(vector_schemas.VectorBase):
    where: Dict[str, Any]
    new_metadata: Dict[str, Any]

class ClearCollectionRequest(vector_schemas.VectorBase):
    pass

class CountRequest(vector_schemas.VectorBase):
    pass

@router.post("/ensure-collection", status_code=200)
def ensure_collection(req: vector_schemas.EnsureCollectionRequest = Body(...)):
    try:
        vector_service.ensure_collection(req.database, req.collection)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add", status_code=201)
def add_vectors(req: vector_schemas.AddRequest = Body(...)):
    try:
        vector_service.add(req)
        return {"status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=vector_schemas.QueryResponse)
def query_vectors(req: vector_schemas.QueryRequest = Body(...)):
    try:
        # The request object `req` now correctly has a `where` attribute
        return vector_service.query(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete", status_code=200)
def delete_vectors(req: vector_schemas.DeleteRequest = Body(...)):
    try:
        # The request object `req` now correctly has a `where` attribute
        vector_service.delete(req)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-metadata", status_code=200)
def update_metadata(req: UpdateMetadataRequest = Body(...)):
    try:
        vector_service.update_metadata(req.collection, req.where, req.new_metadata)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-collection", status_code=200)
def clear_collection(req: ClearCollectionRequest = Body(...)):
    try:
        vector_service.clear_collection(req.collection)
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-all", response_model=vector_schemas.GetAllResponse)
def get_all_vectors(req: vector_schemas.VectorBase = Body(...)):
    try:
        return vector_service.get_all(req.collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/count", response_model=vector_schemas.CountResponse)
def count_vectors(req: CountRequest = Body(...)):
    try:
        count = vector_service.count(req.collection)
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/size", response_model=vector_schemas.SizeResponse)
def get_storage_size():
    try:
        size = vector_service.get_storage_size()
        return {"size_bytes": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))