# backend/app/api/v1/endpoints/knowledge_base.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List
import sqlite3
from app.database import get_db_connection
from app.services import knowledge_graph_service, knowledge_base_service
from app.schemas.proxy_schemas import ApiConfig

router = APIRouter()
logging.basicConfig(level=logging.INFO)

class NotePayload(BaseModel):
    file_path: str
    content: str
    title: str

class ProcessNoteRequest(BaseModel):
    note: NotePayload

class ProcessAllNotesRequest(BaseModel):
    notes: List[NotePayload]

class ProcessFileRequest(BaseModel):
    file_path: str
    content: str
    api_config: ApiConfig

@router.post("/process-file", status_code=200)
async def process_file(payload: ProcessFileRequest, request: Request):
    """
    Receives raw file content, chunks, embeds, and indexes it.
    """
    print(request)
    print(payload)
    try:
        await knowledge_base_service.process_and_embed_file(
            payload.file_path, payload.content, payload.api_config, request
        )
        return {"status": "ok", "message": f"File {payload.file_path} processed successfully."}
    except Exception as e:
        logging.error(f"Failed to process file {payload.file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process_note_links", status_code=200)
async def process_note_links(request: ProcessNoteRequest, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Receives a single note's content, parses for [[WikiLinks]], and updates the note_links table.
    """
    # Log Point 3: Check content received by FastAPI
    logging.info(f"[DEBUG_WIKILINK] FastAPI process_note_links received content (first 50 chars): '{request.note.content[:50]}...'")
    try:
        knowledge_graph_service.update_links_for_note(
            conn=conn,
            note_path=request.note.file_path,
            content=request.note.content
        )
        return {"status": "ok", "message": f"Links processed for {request.note.file_path}"}
    except Exception as e:
        logging.error(f"Failed to process links for {request.note.file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_all_note_links", status_code=200)
async def process_all_note_links(request: ProcessAllNotesRequest, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Receives all notes, clears existing links, and rebuilds the entire knowledge graph.
    """
    try:
        knowledge_graph_service.rebuild_all_links(conn, request.notes)
        return {"status": "ok", "message": "Knowledge graph rebuilt successfully."}
    except Exception as e:
        logging.error(f"Failed to rebuild knowledge graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph-data")
async def get_graph_data(conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Retrieves the entire knowledge graph data (nodes and links).
    """
    try:
        graph_data = knowledge_graph_service.get_graph_data(conn)
        return graph_data
    except Exception as e:
        logging.error(f"Failed to get knowledge graph data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notes/{note_id:path}")
async def get_note_details(note_id: str, conn: sqlite3.Connection = Depends(get_db_connection)):
    """
    Retrieves details for a single note, including backlinks and outgoing links.
    The `:path` converter allows the note_id to contain slashes.
    """
    try:
        note_details = knowledge_graph_service.get_note_details(conn, note_id)
        if not note_details:
            raise HTTPException(status_code=404, detail="Note not found")
        return note_details
    except Exception as e:
        logging.error(f"Failed to get details for note {note_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))