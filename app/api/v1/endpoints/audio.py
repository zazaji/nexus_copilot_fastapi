# backend/app/api/v1/endpoints/audio.py
import logging
import hashlib
import os
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment
import httpx
import json

from app.core.config import settings
from app.schemas.proxy_schemas import ApiConfig
from app.services import shared_services

router = APIRouter()
logger = logging.getLogger(__name__)

TTS_CACHE_DIR = os.path.join(settings.NEXUS_DATA_PATH, "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

class SpeechRequest(BaseModel):
    input: str
    api_config: ApiConfig

@router.post("/speech")
async def text_to_speech(payload: SpeechRequest):
    """
    Generates audio from text using a configured TTS provider, with caching.
    """
    text_hash = hashlib.sha256(payload.input.encode('utf-8')).hexdigest()
    cached_file_path = os.path.join(TTS_CACHE_DIR, f"{text_hash}.mp3")

    if os.path.exists(cached_file_path):
        logger.info(f"Serving cached TTS audio for hash: {text_hash}")
        return FileResponse(cached_file_path, media_type="audio/mpeg")

    tts_assignment = payload.api_config.assignments.tts
    if not tts_assignment:
        raise HTTPException(status_code=400, detail="TTS model is not configured in settings.")

    logger.info(f"No cache found. Generating new TTS audio for hash: {text_hash}")
    shared_services.log_api_call("tts", tts_assignment.modelName)

    tts_provider = next((p for p in payload.api_config.providers if p.id == tts_assignment.providerId), None)
    if not tts_provider:
        raise HTTPException(status_code=400, detail=f"Provider for TTS model not found: {tts_assignment.providerId}")

    api_key = shared_services._select_random_key(tts_provider.apiKey)
    target_url = f"{tts_provider.baseUrl.strip('/')}/audio/speech"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    request_body = {
        "model": tts_assignment.modelName,
        "input": payload.input,
        "voice": "alloy",
        "response_format": "wav"
    }

    client = shared_services.get_client(tts_provider.proxy)
    try:
        logger.info(f"Sending TTS request to {target_url} with body: {json.dumps(request_body)}")
        response = await client.post(target_url, headers=headers, json=request_body, timeout=60.0)
        response.raise_for_status()
        
        wav_data = await response.aread()
        
        wav_audio = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")
        mono_audio = wav_audio.set_channels(1)
        
        mp3_buffer = io.BytesIO()
        mono_audio.export(mp3_buffer, format="mp3", bitrate="64k")
        mp3_data = mp3_buffer.getvalue()

        with open(cached_file_path, "wb") as f:
            f.write(mp3_data)
        
        logger.info(f"Successfully generated and cached TTS audio: {cached_file_path}")
        return FileResponse(cached_file_path, media_type="audio/mpeg")

    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            if "error" in error_json and "message" in error_json["error"]:
                error_detail = error_json["error"]["message"]
        except Exception:
            pass
        logger.error(f"TTS API returned error {e.response.status_code}: {error_detail}")
        logger.error(f"Request body that caused the error: {json.dumps(request_body)}")
        raise HTTPException(
            status_code=502,
            detail=f"External TTS API Error ({e.response.status_code}): {error_detail}. Request Body: {json.dumps(request_body)}"
        )
    except httpx.RequestError as e:
        logger.error(f"Request to TTS API failed: {e}")
        raise HTTPException(status_code=504, detail=f"Network error when contacting TTS service: {e}")
    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process TTS audio: {e}")
    finally:
        if tts_provider.proxy:
            await client.aclose()