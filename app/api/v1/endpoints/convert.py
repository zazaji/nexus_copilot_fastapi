# backend/app/api/v1/endpoints/convert.py
import asyncio
import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
from app.services import parser_service

router = APIRouter()

class ConversionRequest(BaseModel):
    input_dir: str
    output_dir: str

SUPPORTED_EXTENSIONS = {'.ppt', '.pptx', '.doc', '.docx', '.pdf', '.txt'}

async def conversion_streamer(input_dir: str, output_dir: str) -> AsyncGenerator[str, None]:
    """
    Streams the progress of a directory conversion to Markdown.
    """
    files_to_convert = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                files_to_convert.append(os.path.join(root, file))

    total_files = len(files_to_convert)
    if total_files == 0:
        yield f"data: {json.dumps({'progress': 100, 'message': 'No supported files found to convert.'})}\n\n"
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, file_path in enumerate(files_to_convert):
        progress = int(((i + 1) / total_files) * 100)
        file_name = os.path.basename(file_path)
        
        output_filename = os.path.splitext(file_name)[0] + ".md"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            message = f"Skipped (exists): {file_name}"
            yield f"data: {json.dumps({'progress': progress, 'message': message})}\n\n"
            await asyncio.sleep(0.01) # Give a small delay for the UI to update
            continue

        message = f"Converting: {file_name}"
        yield f"data: {json.dumps({'progress': progress, 'message': message})}\n\n"
        
        try:
            # Use a thread pool for synchronous parsing to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, parser_service.parse_file, file_path)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            error_message = f"Error converting {file_name}: {e}"
            yield f"data: {json.dumps({'progress': progress, 'message': error_message, 'error': True})}\n\n"
        
        await asyncio.sleep(0.05) # Give a small delay

    yield f"data: {json.dumps({'progress': 100, 'message': 'Conversion complete!'})}\n\n"


@router.post("/to-markdown")
async def convert_to_markdown(request: ConversionRequest):
    if not os.path.isdir(request.input_dir):
        raise HTTPException(status_code=400, detail="Input path is not a valid directory.")
    
    return StreamingResponse(
        conversion_streamer(request.input_dir, request.output_dir),
        media_type="text/event-stream"
    )