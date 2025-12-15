from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class LandmarkInput(BaseModel):
    """Input for requesting hand landmark extraction from an image or frame."""

    session_id: Optional[str] = Field(
        None,
        description="Client-provided session identifier",
        json_schema_extra={"example": "session-123"},
    )
    user_id: Optional[str] = Field(
        None,
        description="Client-provided user identifier",
        json_schema_extra={"example": "user-abc"},
    )
    image_url: Optional[str] = Field(
        None,
        description="URL to an image or video frame",
        json_schema_extra={"example": "https://example.com/frame.jpg"},
    )
    frame_base64: Optional[str] = Field(
        None,
        description="Base64-encoded image data (data URL format supported)",
        json_schema_extra={"example": "data:image/jpeg;base64,..."},
    )
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Optional client metadata (device, app version, etc.)",
        json_schema_extra={"example": {"device": "iphone"}},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session-123",
                "user_id": "user-abc",
                "image_url": "https://example.com/frame.jpg",
                "metadata": {"device": "iphone"},
            }
        },
        "protected_namespaces": (),
    }


class LandmarkResult(BaseModel):
    """Output for hand landmark extraction."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this landmark extraction",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440100"},
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier from the request",
        json_schema_extra={"example": "session-123"},
    )
    user_id: Optional[str] = Field(
        None,
        description="User identifier from the request",
        json_schema_extra={"example": "user-abc"},
    )
    landmarks: List[List[float]] = Field(
        ...,
        description="21 hand landmarks (x,y)",
        min_items=21,
        max_items=21,
        json_schema_extra={
            "example": [
                [0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                [0.70, 0.40],
            ]
        },
    )
    method: str = Field(
        "stub",
        description='Extraction method used (e.g., "mediapipe" or "stub")',
        json_schema_extra={"example": "mediapipe"},
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of extraction",
        json_schema_extra={"example": "2025-01-16T12:34:56Z"},
    )
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Echo of caller-provided metadata",
        json_schema_extra={"example": {"device": "iphone"}},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440100",
                "session_id": "session-123",
                "user_id": "user-abc",
                "landmarks": [[0.5, 0.6], [0.52, 0.58], [0.54, 0.56]],
                "method": "mediapipe",
                "extracted_at": "2025-01-16T12:34:56Z",
                "metadata": {"device": "iphone"},
            }
        },
        "protected_namespaces": (),
    }
