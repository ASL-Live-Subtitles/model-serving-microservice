from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class GlossPredictionInput(BaseModel):
    """Input for predicting ASL glosses from landmarks."""

    session_id: Optional[str] = Field(
        None,
        description="Session identifier",
        json_schema_extra={"example": "session-123"},
    )
    user_id: Optional[str] = Field(
        None,
        description="User identifier",
        json_schema_extra={"example": "user-abc"},
    )
    landmark_id: Optional[UUID] = Field(
        None,
        description="Reference to a previously extracted landmark record",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440100"},
    )
    landmarks: Optional[List[List[float]]] = Field(
        None,
        description="Inline 21-point landmarks if not referencing an existing record",
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session-123",
                "landmark_id": "550e8400-e29b-41d4-a716-446655440100",
                "landmarks": None,
            }
        },
        "protected_namespaces": (),
    }


class GlossPredictionResult(BaseModel):
    """Output for gloss predictions from landmarks."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this gloss prediction",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440200"},
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier",
        json_schema_extra={"example": "session-123"},
    )
    user_id: Optional[str] = Field(
        None,
        description="User identifier",
        json_schema_extra={"example": "user-abc"},
    )
    landmark_id: Optional[UUID] = Field(
        None,
        description="Reference to a landmark record used for this prediction",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440100"},
    )
    gloss: List[str] = Field(
        ...,
        description="Predicted ASL gloss tokens (prefix removed)",
        json_schema_extra={"example": ["HELLO"]},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction",
        json_schema_extra={"example": 0.93},
    )
    inference_method: str = Field(
        "stub",
        description='Method used for inference (e.g., "openai", "heuristic", "stub")',
        json_schema_extra={"example": "stub"},
    )
    model_version: str = Field(
        "v1.0.0",
        description="Version of the model or prompt used",
        json_schema_extra={"example": "v1.0.0"},
    )
    raw_openai_response: Optional[dict] = Field(
        None,
        description="Raw OpenAI response payload (if OpenAI was used)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the prediction was created",
        json_schema_extra={"example": "2025-01-16T12:34:56Z"},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440200",
                "session_id": "session-123",
                "landmark_id": "550e8400-e29b-41d4-a716-446655440100",
                "gloss": ["HELLO"],
                "confidence": 0.93,
                "inference_method": "stub",
                "model_version": "v1.0.0",
                "raw_openai_response": {"choices": []},
                "created_at": "2025-01-16T12:34:56Z",
            }
        },
        "protected_namespaces": (),
    }
