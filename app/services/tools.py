import httpx
import logging
import json
import psutil
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)

def _render_template(template: Any, params: Dict[str, Any]) -> Any:
    """
    Recursively renders a template (string, dict, or list) with given parameters.
    """
    if isinstance(template, str):
        for key, value in params.items():
            placeholder = f"{{{{{key}}}}}"
            # Ensure value is a string for replacement
            str_value = str(value) if value is not None else ""
            template = template.replace(placeholder, str_value)
        return template
    elif isinstance(template, dict):
        return {k: _render_template(v, params) for k, v in template.items()}
    elif isinstance(template, list):
        return [_render_template(item, params) for item in template]
    else:
        return template

async def execute_webhook_tool(
    url: str,
    method: str,
    headers: Optional[Dict[str, str]],
    body_template: Optional[Any],
    params: Dict[str, Any]
):
    """
    Executes a tool of type 'webhook' with customizable method, headers, and body.
    """
    logging.info(f"Executing webhook tool. Method: {method}, URL: {url}")

    # Render headers and body with runtime parameters
    final_headers = _render_template(headers or {}, params)
    final_body = _render_template(body_template or {}, params)

    logging.info(f"Final Headers: {final_headers}")
    logging.info(f"Final Body: {final_body}")

    try:
        async with httpx.AsyncClient() as client:
            request = client.build_request(
                method=method.upper(),
                url=url,
                headers=final_headers,
                json=final_body if method.upper() in ["POST", "PUT", "PATCH"] else None,
                params=final_body if method.upper() == "GET" else None,
                timeout=30.0
            )
            response = await client.send(request)
            response.raise_for_status()

            return {
                "status_code": response.status_code,
                "response_text": response.text[:1000] # Truncate long responses
            }
    except httpx.RequestError as e:
        logging.error(f"Request to webhook URL {url} failed: {e}")
        raise Exception(f"Network error calling webhook: {e}")
    except httpx.HTTPStatusError as e:
        logging.error(f"Webhook URL {url} returned error status {e.response.status_code}: {e.response.text}")
        raise Exception(f"Webhook returned error: {e.response.status_code} - {e.response.text}")

async def get_memory_usage() -> Dict[str, Any]:
    """
    Gets the current system RAM and swap memory usage.
    """
    logging.info("Executing tool: get_memory_usage")
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    def bytes_to_gb(b):
        return round(b / (1024**3), 2)

    usage = {
        "ram_total_gb": bytes_to_gb(ram.total),
        "ram_used_gb": bytes_to_gb(ram.used),
        "ram_percent": ram.percent,
        "swap_total_gb": bytes_to_gb(swap.total),
        "swap_used_gb": bytes_to_gb(swap.used),
        "swap_percent": swap.percent
    }
    return usage