"""
Thin HTTP client the worker uses to call the inference layer (Phase 2).

Enforces the brief's hard SLA directly in code: "model inference must
return a result in under 2.0 seconds." We set the HTTP timeout to exactly
that budget — if the model orchestrator can't answer in time, we fail
fast rather than let a slow model hang the worker's flow-controlled
concurrency slot (which would otherwise create backpressure all the way
to the Pub/Sub queue).
"""
import base64
import time

import requests

from app.config import get_settings
from app.schemas import CropType, InferenceResult

settings = get_settings()


class InferenceTimeout(Exception):
    pass


def run_inference(*, crop_type: CropType, image_bytes: bytes) -> InferenceResult:
    start = time.monotonic()
    try:
        response = requests.post(
            f"{settings.inference_service_url}/infer/{crop_type.value}",
            json={"image_b64": base64.b64encode(image_bytes).decode("ascii")},
            timeout=settings.inference_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except requests.Timeout as exc:
        raise InferenceTimeout(
            f"inference exceeded {settings.inference_timeout_seconds}s SLA"
        ) from exc

    latency_ms = (time.monotonic() - start) * 1000
    return InferenceResult(
        predicted_class=body["predicted_class"],
        confidence=body["confidence"],
        model_version=body["model_version"],
        latency_ms=latency_ms,
        crop_type=crop_type,
    )
