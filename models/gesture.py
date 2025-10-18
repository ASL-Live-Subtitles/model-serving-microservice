from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

# -------------------------------
# INPUT models
# -------------------------------

class GestureInput(BaseModel):
    """Input model for gesture recognition requests."""
    
    landmarks: List[List[float]] = Field(
        ...,
        description="List of hand landmark coordinates (21 points with x,y coordinates each)",
        min_items=21,
        max_items=21,
        json_schema_extra={
            "example": [
                [0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                [0.70, 0.40]
            ]
        }
    )
    
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking predictions",
        json_schema_extra={"example": "user123"}
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "landmarks": [
                    [0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                    [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                    [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                    [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                    [0.70, 0.40]
                ],
                "user_id": "user123"
            }
        }
    }

class ModelInput(BaseModel):
    """Input model for registering new ML models."""
    
    name: str = Field(
        ...,
        description="Human-readable name of the model",
        json_schema_extra={"example": "ASL Keypoint Classifier v2"}
    )
    
    version: str = Field(
        ...,
        description="Version identifier for the model",
        json_schema_extra={"example": "v2.1.0"}
    )
    
    description: Optional[str] = Field(
        None,
        description="Detailed description of the model",
        json_schema_extra={"example": "Improved TensorFlow Lite model for ASL gesture recognition with 95% accuracy"}
    )
    
    model_path: str = Field(
        ...,
        description="File path or URL to the model file",
        json_schema_extra={"example": "/models/asl_classifier_v2.tflite"}
    )
    
    input_shape: List[int] = Field(
        ...,
        description="Expected input shape for the model",
        json_schema_extra={"example": [42]}
    )
    
    output_classes: int = Field(
        ...,
        description="Number of output classes the model can predict",
        json_schema_extra={"example": 37}
    )
    
    model_type: str = Field(
        "classification",
        description="Type of ML model (classification, regression, etc.)",
        json_schema_extra={"example": "classification"}
    )

# -------------------------------
# OUTPUT models
# -------------------------------

class GestureResult(BaseModel):
    """Output model for gesture recognition results."""
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this prediction result",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"}
    )
    
    landmarks: List[List[float]] = Field(
        ...,
        description="Original hand landmark coordinates used for prediction",
        json_schema_extra={
            "example": [
                [0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                [0.70, 0.40]
            ]
        }
    )
    
    predicted_gesture: str = Field(
        ...,
        description="The predicted ASL gesture (A-Z, 0-9, or special characters)",
        json_schema_extra={"example": "A"}
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction (0.0-1.0)",
        json_schema_extra={"example": 0.95}
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process the prediction in milliseconds",
        json_schema_extra={"example": 15.2}
    )
    
    model_version: str = Field(
        ...,
        description="Version of the model used for this prediction",
        json_schema_extra={"example": "v1.0.0"}
    )
    
    predicted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the prediction was made",
        json_schema_extra={"example": "2025-01-16T12:34:56Z"}
    )
    
    user_id: Optional[str] = Field(
        None,
        description="User identifier if provided in the request",
        json_schema_extra={"example": "user123"}
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "landmarks": [[0.5, 0.6], [0.52, 0.58]],
                "predicted_gesture": "A",
                "confidence": 0.95,
                "processing_time_ms": 15.2,
                "model_version": "v1.0.0",
                "predicted_at": "2025-01-16T12:34:56Z",
                "user_id": "user123"
            }
        }
    }

class ModelInfo(BaseModel):
    """Output model for ML model information."""
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this model",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440001"}
    )
    
    name: str = Field(
        ...,
        description="Human-readable name of the model",
        json_schema_extra={"example": "ASL Keypoint Classifier"}
    )
    
    version: str = Field(
        ...,
        description="Version identifier for the model",
        json_schema_extra={"example": "v1.0.0"}
    )
    
    description: Optional[str] = Field(
        None,
        description="Detailed description of the model",
        json_schema_extra={"example": "TensorFlow Lite model for ASL gesture recognition"}
    )
    
    input_shape: List[int] = Field(
        ...,
        description="Expected input shape for the model",
        json_schema_extra={"example": [42]}
    )
    
    output_classes: int = Field(
        ...,
        description="Number of output classes the model can predict",
        json_schema_extra={"example": 37}
    )
    
    model_type: str = Field(
        ...,
        description="Type of ML model",
        json_schema_extra={"example": "classification"}
    )
    
    accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model accuracy score if available",
        json_schema_extra={"example": 0.94}
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the model was registered",
        json_schema_extra={"example": "2025-01-16T10:00:00Z"}
    )
    
    is_active: bool = Field(
        True,
        description="Whether this model is currently active for predictions",
        json_schema_extra={"example": True}
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "name": "ASL Keypoint Classifier",
                "version": "v1.0.0",
                "description": "TensorFlow Lite model for ASL gesture recognition",
                "input_shape": [42],
                "output_classes": 37,
                "model_type": "classification",
                "accuracy": 0.94,
                "created_at": "2025-01-16T10:00:00Z",
                "is_active": True
            }
        }
    }

class PredictionRequest(BaseModel):
    """Model for batch prediction requests."""
    
    id: Optional[UUID] = Field(
        None,
        description="Unique identifier for this batch request",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440002"}
    )
    
    batch_name: str = Field(
        ...,
        description="Name for this batch prediction request",
        json_schema_extra={"example": "Daily ASL Recognition Batch"}
    )
    
    model_id: UUID = Field(
        ...,
        description="ID of the model to use for predictions",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440001"}
    )
    
    input_data: List[GestureInput] = Field(
        ...,
        description="List of gesture inputs to process",
        min_items=1,
        json_schema_extra={
            "example": [
                {
                    "landmarks": [[0.5, 0.6], [0.52, 0.58]],
                    "user_id": "user123"
                }
            ]
        }
    )
    
    status: str = Field(
        "pending",
        description="Status of the batch request (pending, processing, completed, failed)",
        json_schema_extra={"example": "pending"}
    )
    
    created_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the batch request was created",
        json_schema_extra={"example": "2025-01-16T11:00:00Z"}
    )
    
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the batch request was completed",
        json_schema_extra={"example": "2025-01-16T11:05:00Z"}
    )
    
    results: Optional[List[GestureResult]] = Field(
        None,
        description="Results of the batch prediction (populated when completed)",
        json_schema_extra={"example": []}
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440002",
                "batch_name": "Daily ASL Recognition Batch",
                "model_id": "550e8400-e29b-41d4-a716-446655440001",
                "input_data": [
                    {
                        "landmarks": [[0.5, 0.6], [0.52, 0.58]],
                        "user_id": "user123"
                    }
                ],
                "status": "pending",
                "created_at": "2025-01-16T11:00:00Z",
                "completed_at": None,
                "results": None
            }
        }
    }