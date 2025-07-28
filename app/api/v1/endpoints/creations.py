# backend/app/api/v1/endpoints/creations.py
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse,JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, AsyncGenerator
import httpx
import json
import asyncio
import time

from app.schemas.proxy_schemas import ApiProvider
from app.services import shared_services

router = APIRouter()
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    creation_type: str = Field(..., alias="creationType")
    prompt: str
    model_name: str = Field(..., alias="modelName")
    params: Dict[str, Any]
    provider: ApiProvider

async def generate_image(prompt: str, model_name: str, params: Dict[str, Any], provider: ApiProvider):
    shared_services.log_api_call("image_gen", model_name)

    api_key = shared_services._select_random_key(provider.apiKey)
    target_url = f"{provider.baseUrl.strip('/')}/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    request_body = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": params.get("size", "1024x1024")
    }

    client = shared_services.get_client(provider.proxy)
    try:
        logger.info(f"Sending image generation request to {target_url} with body: {json.dumps(request_body)}")
        response = await client.post(target_url, headers=headers, json=request_body, timeout=120.0)
        response.raise_for_status()
        
        response_data = response.json()
        logger.info(f"Received successful response from image generation API: {response_data}")
        image_url = response_data["data"][0]["url"]
        revised_prompt = response_data["data"][0].get("revised_prompt", prompt)
        
        return {
            "url": image_url,
            "prompt": revised_prompt,
            "content_type": "image/png"
        }
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            if "error" in error_json and "message" in error_json["error"]:
                error_detail = error_json["error"]["message"]
        except Exception:
            pass
        logger.error(f"Image generation API returned error {e.response.status_code}: {error_detail}")
        logger.error(f"Request body that caused the error: {json.dumps(request_body)}")
        raise HTTPException(
            status_code=502,
            detail=f"External API Error ({e.response.status_code}): {error_detail}. Request Body: {json.dumps(request_body)}"
        )
    except httpx.RequestError as e:
        logger.error(f"Request to image generation API failed: {e}")
        raise HTTPException(status_code=504, detail=f"Network error when contacting image service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during image generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during image generation: {e}")
    finally:
        if provider.proxy:
            await client.aclose()


async def generate_audio(prompt: str, model_name: str, params: Dict[str, Any], provider: ApiProvider):
    shared_services.log_api_call("tts", model_name)

    api_key = shared_services._select_random_key(provider.apiKey)
    target_url = f"{provider.baseUrl.strip('/')}/audio/speech"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    request_body = {
        "model": model_name,
        "input": prompt,
        "voice": params.get("voice", "alloy"),
    }

    client = shared_services.get_client(provider.proxy)
    try:
        logger.info(f"Sending audio generation request to {target_url} with body: {json.dumps(request_body)}")
        async with client.stream('POST', target_url, headers=headers, json=request_body, timeout=60.0) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        logger.error(f"Audio generation API returned error {e.response.status_code}: {error_detail}")
        logger.error(f"Request body that caused the error: {json.dumps(request_body)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio generation stream: {e}", exc_info=True)
    finally:
        if provider.proxy:
            await client.aclose()

async def generate_video(prompt: str, model_name: str, params: Dict[str, Any], provider: ApiProvider) -> AsyncGenerator[str, None]:
    """
    Handles the asynchronous workflow for video generation using Server-Sent Events.
    """
    shared_services.log_api_call("video_gen", model_name)
    api_key = shared_services._select_random_key(provider.apiKey)
    client = shared_services.get_client(provider.proxy)
    
    try:
        # Step 1: Initiate video generation task
        initiate_url = f"{provider.baseUrl.strip('/')}/videos/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        initiate_body = {"model": model_name, "prompt": prompt}
        
        logger.info(f"Initiating video generation at {initiate_url}")
        initiate_response = await client.post(initiate_url, headers=headers, json=initiate_body, timeout=30.0)
        initiate_response.raise_for_status()
        task_data = initiate_response.json()
        task_id = task_data.get("id")

        if not task_id:
            raise Exception("Video generation API did not return a task ID.")
        
        logger.info(f"Video generation task started with ID: {task_id}")
        yield f"data: {json.dumps({'status': 'PENDING', 'message': 'Task submitted...', 'progress': 5})}\n\n"

        # Step 2: Poll for task status and stream updates
        status_url = f"{initiate_url}/../tasks/{task_id}"
        POLLING_INTERVAL_SECONDS = 3
        MAX_POLLING_SECONDS = 600
        start_time = time.time()

        while time.time() - start_time < MAX_POLLING_SECONDS:
            await asyncio.sleep(POLLING_INTERVAL_SECONDS)
            status_response = await client.get(status_url, headers=headers, timeout=30.0)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            task_status = status_data.get("task_status")
            progress = status_data.get("task_progress", 10) # Use a default progress if not provided
            
            yield f"data: {json.dumps({'status': task_status, 'message': f'Processing... ({progress}%)', 'progress': progress})}\n\n"

            if task_status == "SUCCESS":
                video_result = status_data.get("video_result", [])
                if not video_result:
                    raise Exception("Video task succeeded but returned no video data.")
                
                video_url = video_result[0].get("url")
                if not video_url:
                    raise Exception("Video task succeeded but response is missing the video URL.")

                final_payload = {
                    "status": "SUCCESS",
                    "url": video_url,
                    "prompt": prompt,
                    "content_type": "video/mp4"
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                return # End the stream
            
            elif task_status == "FAILED":
                error_detail = status_data.get("error", "Video generation failed with an unknown error.")
                raise Exception(f"Video generation failed: {error_detail}")

        raise Exception("Video generation timed out after 10 minutes.")

    except Exception as e:
        logger.error(f"An error occurred during video generation stream: {e}", exc_info=True)
        error_payload = {"status": "FAILED", "message": str(e)}
        yield f"data: {json.dumps(error_payload)}\n\n"
    finally:
        if provider.proxy:
            await client.aclose()


@router.post("/generate")
async def generate_creation(payload: GenerationRequest):
    """
    Unified endpoint for generating creative artifacts.
    """
    if payload.creation_type == "image":
        result = await generate_image(payload.prompt, payload.model_name, payload.params, payload.provider)
        return JSONResponse(content=result)
    elif payload.creation_type == "audio":
        return StreamingResponse(generate_audio(payload.prompt, payload.model_name, payload.params, payload.provider), media_type="audio/mpeg")
    elif payload.creation_type == "video":
        return StreamingResponse(generate_video(payload.prompt, payload.model_name, payload.params, payload.provider), media_type="text/event-stream")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported creation type: {payload.creation_type}")