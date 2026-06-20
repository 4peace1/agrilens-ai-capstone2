"""
TFLite model-serving layer (Phase 2: "Optimized Inference Layer with
TFLite & Triton").

Design choices that directly target the brief's constraints:
  - Models are loaded once at process startup and held in memory as
    `tf.lite.Interpreter` instances — no per-request disk I/O or model
    re-load, which is what makes the sub-2.0s SLA achievable.
  - This process does ONLY inference — no HTTP-traffic-handling /
    business logic — satisfying "Don't use a single large container for
    both the API and the ML model." It's deployed as its own GKE
    Deployment (see k8s/inference-deployment.yaml) with its own HPA.
  - Cassava and Cocoa get *separate* interpreter instances (and in
    production, separate pods — see ab_router.py) so a burst of cassava
    traffic during planting season can scale independently of cocoa
    traffic, rather than contending for the same model's lock.
  - Numpy/TFLite calls release the GIL during `invoke()`, so a single
    worker process can still serve concurrent requests without one slow
    inference blocking another.
"""
import base64
import io
import logging
import os
import time
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from inference.model_server.ab_router import select_variant

try:
    # Production: the lightweight tflite-runtime package (no full TF dep).
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # local dev fallback if only full tensorflow is installed
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrilens.inference")

app = FastAPI(title="AgriLens AI — Inference Orchestrator")

MODEL_PATHS: Dict[str, str] = {
    "cassava-v3-stable": "models/cassava_v3_int8.tflite",
    "cassava-v4-canary": "models/cassava_v4_int8.tflite",
    "cocoa-v2-stable": "models/cocoa_v2_int8.tflite",
    "maize-v1-stable": "models/maize_v1_int8.tflite",
}

CLASS_LABELS: Dict[str, list[str]] = {
    "cassava-v3-stable": ["healthy", "mosaic_disease", "brown_streak", "bacterial_blight"],
    "cassava-v4-canary": ["healthy", "mosaic_disease", "brown_streak", "bacterial_blight"],
    "cocoa-v2-stable": ["healthy", "black_pod_rot", "frosty_pod_rot", "swollen_shoot"],
    "maize-v1-stable": ["healthy", "rust", "gray_leaf_spot", "blight"],
}

_interpreters: Dict[str, Interpreter] = {}


def _crop_filter() -> set[str] | None:
    """When set (e.g. CROP_FILTER=cassava), this pod only loads model
    variants for that crop, matching the architecture's split between
    dedicated Cassava Model Pods and Cocoa Model Pods — each crop scales
    independently and a crash/OOM in one model family can't take down
    the other. Unset (the local-dev default) loads everything in one
    process for convenience."""
    raw = os.environ.get("CROP_FILTER")
    return {c.strip() for c in raw.split(",")} if raw else None


@app.on_event("startup")
def load_models() -> None:
    crop_filter = _crop_filter()
    for version, path in MODEL_PATHS.items():
        crop = version.split("-")[0]
        if crop_filter and crop not in crop_filter:
            continue
        try:
            interpreter = Interpreter(model_path=path, num_threads=2)
            interpreter.allocate_tensors()
            _interpreters[version] = interpreter
            logger.info("loaded model %s from %s", version, path)
        except (FileNotFoundError, ValueError) as exc:
            # In this reference implementation the .tflite binaries aren't
            # bundled (they're produced by quantize.py against real
            # training data). We log and continue so the service still
            # boots for local development / code review.
            logger.warning("model artifact missing for %s (%s) — run "
                            "inference/quantize.py to produce it", version, exc)


class InferRequest(BaseModel):
    image_b64: str


class InferResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_version: str


def _preprocess(image_bytes: bytes, input_shape) -> np.ndarray:
    _, height, width, channels = input_shape
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize((width, height))
        array = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@app.post("/infer/{crop_type}", response_model=InferResponse)
def infer(crop_type: str, payload: InferRequest):
    start = time.monotonic()

    model_version = select_variant(crop_type, routing_key=payload.image_b64[:32])
    interpreter = _interpreters.get(model_version)
    if interpreter is None:
        raise HTTPException(
            status_code=503,
            detail=f"model {model_version} not loaded (artifact missing)",
        )

    image_bytes = base64.b64decode(payload.image_b64)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensor = _preprocess(image_bytes, input_details[0]["shape"])
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    labels = CLASS_LABELS[model_version]
    best_idx = int(np.argmax(output))
    confidence = float(output[best_idx])

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "crop=%s model=%s class=%s conf=%.3f latency_ms=%.1f",
        crop_type, model_version, labels[best_idx], confidence, elapsed_ms,
    )

    return InferResponse(
        predicted_class=labels[best_idx],
        confidence=confidence,
        model_version=model_version,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "models_loaded": list(_interpreters.keys())}
