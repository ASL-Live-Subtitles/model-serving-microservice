from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class VideoGlossResponse(BaseModel):
    """Response payload for video-based gloss prediction."""

    id: str = Field(..., description="Unique identifier for this video gloss request (DB primary key)")
    gloss: List[str] = Field(..., description="Predicted ASL gloss tokens (prefix removed)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    frames_processed: int = Field(..., description="Number of frames used")
    inference_method: str = Field(..., description="Method used (openai/heuristic/stub)")
    model_version: str = Field("v1.0.0", description="Model version used")
    raw_openai_response: Optional[dict] = Field(
        None,
        description="Raw OpenAI response payload (if OpenAI was used)",
    )


class VideoGlossUpdate(BaseModel):
    """Update payload for an existing video gloss record."""

    gloss: Optional[List[str]] = Field(None, description="Updated gloss tokens")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    inference_method: Optional[str] = Field(None, description="openai/fallback/stub")
    raw_openai_response: Optional[dict] = Field(None, description="Raw OpenAI response")
    detected: Optional[bool] = Field(None, description="Whether a hand was detected")
    landmarks: Optional[List[List[List[float]]]] = Field(
        None,
        description="Optional updated landmark frames",
    )
