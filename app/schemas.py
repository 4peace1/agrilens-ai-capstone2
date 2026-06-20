"""
Pydantic schemas shared across the gateway, worker, and inference layer.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DiagnosticStatus(str, Enum):
    """Explicit status machine for a diagnostic request.

    PENDING     -> record created, signed upload URL issued, image not yet
                   received in GCS.
    UPLOADED    -> client confirmed the raw image PUT succeeded.
    QUEUED      -> a Pub/Sub message has been published for processing.
    PROCESSING  -> the worker has picked up the message and is normalizing
                   the image / running inference.
    COMPLETED   -> inference result is available and has been persisted.
    FAILED      -> terminal failure (bad image, model error, timeout, etc).
    """

    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CropType(str, Enum):
    CASSAVA = "cassava"
    COCOA = "cocoa"
    MAIZE = "maize"  # third target crop referenced in the brief


class DiagnosticCreateRequest(BaseModel):
    """Payload sent by the mobile app when it has a photo ready to submit.

    `captured_at` is deliberately a free-standing client-supplied timestamp
    (not server `now()`), because the brief requires offline-first support:
    a farmer may capture a photo with zero connectivity and only sync hours
    or days later. The server must accept and honor that historical time.
    """

    crop_type: CropType
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    captured_at: datetime
    image_content_type: str = Field(
        default="image/jpeg",
        description="MIME type of the image to be uploaded via the signed URL.",
    )
    farmer_device_id: str = Field(
        ..., description="Pseudonymous device/installation id, never raw farmer PII."
    )

    @field_validator("image_content_type")
    @classmethod
    def _validate_mime(cls, v: str) -> str:
        allowed = {"image/jpeg", "image/png", "image/webp"}
        if v not in allowed:
            raise ValueError(f"image_content_type must be one of {allowed}")
        return v


class DiagnosticCreateResponse(BaseModel):
    diagnostic_id: UUID
    upload_url: str
    gcs_uri: str
    status: DiagnosticStatus
    expires_in_minutes: int


class DiagnosticCompleteRequest(BaseModel):
    """Sent by the client once the signed-URL PUT to GCS has finished."""

    diagnostic_id: UUID


class DiagnosticStatusResponse(BaseModel):
    diagnostic_id: UUID
    status: DiagnosticStatus
    crop_type: CropType
    masked_latitude: float
    masked_longitude: float
    captured_at: datetime
    result: Optional["InferenceResult"] = None
    error_message: Optional[str] = None


class InferenceResult(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    latency_ms: float
    crop_type: CropType


class ProcessingMessage(BaseModel):
    """Body of the Pub/Sub message published by the gateway and consumed
    by the image-processing worker."""

    diagnostic_id: UUID = Field(default_factory=uuid4)
    correlation_id: str
    raw_gcs_uri: str
    crop_type: CropType
    latitude: float
    longitude: float
    captured_at: datetime
    attempt: int = 1
