from fastapi import FastAPI, HTTPException, status
from typing import Dict, List, Any
from uuid import UUID, uuid4
from datetime import datetime

# Import our models
from models.gesture import (
    GestureInput, GestureResult, ModelInput, ModelInfo, 
    PredictionRequest
)

import os
from dotenv import load_dotenv
load_dotenv()  # so DB_* vars from .env are available

from fastapi import Depends
from db.model_serving_service import (
    ModelsMySQLService, GesturesMySQLService, PredictionsMySQLService
)

def get_models_service(): return ModelsMySQLService()
def get_gestures_service(): return GesturesMySQLService()
def get_predictions_service(): return PredictionsMySQLService()

# Initialize FastAPI app with OpenAPI documentation
app = FastAPI(
    title="Model Serving Microservice",
    description="A microservice for serving machine learning models for hand gesture recognition",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "gestures",
            "description": "Operations for gesture recognition and results management"
        },
        {
            "name": "models",
            "description": "Operations for ML model management and registration"
        },
        {
            "name": "predictions",
            "description": "Operations for batch prediction requests and processing"
        },
        {
            "name": "health",
            "description": "Health check and system status endpoints"
        }
    ]
)

# In-memory storage for demonstration (replace with actual database in production)
gestures_db: Dict[UUID, GestureResult] = {}
models_db: Dict[UUID, ModelInfo] = {}
predictions_db: Dict[UUID, PredictionRequest] = {}

# Dummy helper functions
def make_dummy_gesture(gesture_id: UUID = None) -> GestureResult:
    """Create a dummy gesture result for demonstration."""
    return GestureResult(
        id=gesture_id or uuid4(),
        landmarks=[[0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                  [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                  [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                  [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                  [0.70, 0.40]],
        predicted_gesture="A",
        confidence=0.95,
        processing_time_ms=15.2,
        model_version="v1.0.0",
        user_id="demo_user"
    )

def make_dummy_model(model_id: UUID = None) -> ModelInfo:
    """Create a dummy model info for demonstration."""
    return ModelInfo(
        id=model_id or uuid4(),
        name="ASL Keypoint Classifier",
        version="v1.0.0",
        description="TensorFlow Lite model for ASL gesture recognition",
        input_shape=[42],
        output_classes=37,
        model_type="classification",
        accuracy=0.94,
        is_active=True
    )

def make_dummy_prediction(prediction_id: UUID = None) -> PredictionRequest:
    """Create a dummy prediction request for demonstration."""
    dummy_input = GestureInput(
        landmarks=[[0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                  [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                  [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                  [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                  [0.70, 0.40]],
        user_id="demo_user"
    )
    
    return PredictionRequest(
        id=prediction_id or uuid4(),
        batch_name="Demo Batch Prediction",
        model_id=uuid4(),
        input_data=[dummy_input],
        status="completed",
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        results=[make_dummy_gesture()]
    )

# -----------------------------------------------------------------------------
# Gesture Recognition Routes
# -----------------------------------------------------------------------------

# @app.get("/gestures", response_model=List[GestureResult], tags=["gestures"])
# async def get_all_gestures():
#     """
#     Retrieve all gesture recognition results.
#     """
#     if not gestures_db:
#         # Return some dummy data for demonstration
#         dummy_gesture = make_dummy_gesture()
#         gestures_db[dummy_gesture.id] = dummy_gesture
    
#     return list(gestures_db.values())

# @app.get("/gestures/{gesture_id}", response_model=GestureResult, tags=["gestures"])
# async def get_gesture(gesture_id: UUID):
#     """
#     Retrieve a specific gesture recognition result by ID.
#     """
#     if gesture_id not in gestures_db:
#         # Create dummy data if not found
#         dummy_gesture = make_dummy_gesture(gesture_id)
#         gestures_db[gesture_id] = dummy_gesture
    
#     return gestures_db[gesture_id]

# @app.post("/gestures", response_model=GestureResult, status_code=status.HTTP_201_CREATED, tags=["gestures"])
# async def create_gesture_prediction(gesture_input: GestureInput):
#     """
#     Process hand landmarks and return gesture prediction.
#     """
#     # In production, this would call the actual ML model
#     # For now, return dummy prediction
#     result = GestureResult(
#         landmarks=gesture_input.landmarks,
#         predicted_gesture="A",  # Dummy prediction
#         confidence=0.95,
#         processing_time_ms=15.2,
#         model_version="v1.0.0",
#         user_id=gesture_input.user_id
#     )
    
#     gestures_db[result.id] = result
#     return result

# @app.put("/gestures/{gesture_id}", response_model=GestureResult, tags=["gestures"])
# async def update_gesture(gesture_id: UUID, gesture_input: GestureInput):
#     """
#     Update a gesture recognition result (NOT IMPLEMENTED).
#     """
#     raise HTTPException(
#         status_code=status.HTTP_501_NOT_IMPLEMENTED,
#         detail="NOT IMPLEMENTED: Gesture update functionality not yet available"
#     )

# @app.delete("/gestures/{gesture_id}", tags=["gestures"])
# async def delete_gesture(gesture_id: UUID):
#     """
#     Delete a gesture recognition result.
#     """
#     if gesture_id in gestures_db:
#         del gestures_db[gesture_id]
#         return {"message": f"Gesture {gesture_id} deleted successfully"}
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Gesture with ID {gesture_id} not found"
#         )

from fastapi import Body

# List
@app.get("/gestures", tags=["gestures"])
def list_gestures(
    user_id: str | None = None,
    limit: int = 100,
    svc: GesturesMySQLService = Depends(get_gestures_service),
):
    return svc.retrieve(user_id=user_id, limit=limit)

# Get one
@app.get("/gestures/{gesture_id}", tags=["gestures"])
def get_gesture(gesture_id: int, svc: GesturesMySQLService = Depends(get_gestures_service)):
    row = svc.get_one(gesture_id)
    if not row:
        raise HTTPException(404, "Gesture not found")
    # If you want to return parsed JSON for landmarks/probs:
    for col in ("landmarks", "probs"):
        if row.get(col):
            try:
                row[col] = json.loads(row[col])
            except Exception:
                pass
    return row

# Create
@app.post("/gestures", tags=["gestures"], status_code=status.HTTP_201_CREATED)
def create_gesture(
    gesture_input: GestureInput,
    svc: GesturesMySQLService = Depends(get_gestures_service),
):
    if len(gesture_input.landmarks) < 21:
        raise HTTPException(400, "landmarks must have at least 21 points")

    gid = svc.create({
        "landmarks": gesture_input.landmarks,
        "user_id": gesture_input.user_id,
        "source": "api",
    })
    return {"gesture_id": gid}

# Update (either landmarks OR inference)
@app.put("/gestures/{gesture_id}", tags=["gestures"])
def update_gesture(
    gesture_id: int,
    payload: dict = Body(...),  # e.g., {"landmarks":[...]} OR {"predicted_label":"A", "confidence":0.95, ...}
    svc: GesturesMySQLService = Depends(get_gestures_service),
):
    ok = svc.update(gesture_id, payload)
    if not ok:
        raise HTTPException(404, "Gesture not found or nothing to update")
    return {"updated": True}

# Delete
@app.delete("/gestures/{gesture_id}", tags=["gestures"])
def delete_gesture(gesture_id: int, svc: GesturesMySQLService = Depends(get_gestures_service)):
    ok = svc.delete(gesture_id)
    if not ok:
        raise HTTPException(404, "Gesture not found")
    return {"deleted": True}


# -----------------------------------------------------------------------------
# Model Management Routes
# -----------------------------------------------------------------------------

# @app.get("/models", response_model=List[ModelInfo], tags=["models"])
# async def get_all_models():
#     """
#     Retrieve all registered ML models.
#     """
#     if not models_db:
#         # Return some dummy data for demonstration
#         dummy_model = make_dummy_model()
#         models_db[dummy_model.id] = dummy_model
    
#     return list(models_db.values())

# @app.get("/models/{model_id}", response_model=ModelInfo, tags=["models"])
# async def get_model(model_id: UUID):
#     """
#     Retrieve a specific ML model by ID.
#     """
#     if model_id not in models_db:
#         # Create dummy data if not found
#         dummy_model = make_dummy_model(model_id)
#         models_db[model_id] = dummy_model
    
#     return models_db[model_id]

# @app.post("/models", response_model=ModelInfo, status_code=status.HTTP_201_CREATED, tags=["models"])
# async def register_model(model_input: ModelInput):
#     """
#     Register a new ML model.
#     """
#     model_info = ModelInfo(
#         name=model_input.name,
#         version=model_input.version,
#         description=model_input.description,
#         input_shape=model_input.input_shape,
#         output_classes=model_input.output_classes,
#         model_type=model_input.model_type
#     )
    
#     models_db[model_info.id] = model_info
#     return model_info

# @app.put("/models/{model_id}", response_model=ModelInfo, tags=["models"])
# async def update_model(model_id: UUID, model_input: ModelInput):
#     """
#     Update an existing ML model (NOT IMPLEMENTED).
#     """
#     raise HTTPException(
#         status_code=status.HTTP_501_NOT_IMPLEMENTED,
#         detail="NOT IMPLEMENTED: Model update functionality not yet available"
#     )

# @app.delete("/models/{model_id}", tags=["models"])
# async def delete_model(model_id: UUID):
#     """
#     Delete an ML model registration.
#     """
#     if model_id in models_db:
#         del models_db[model_id]
#         return {"message": f"Model {model_id} deleted successfully"}
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Model with ID {model_id} not found"
#         )

@app.get("/models", tags=["models"])
def get_all_models(svc: ModelsMySQLService = Depends(get_models_service)):
    return svc.retrieve()

@app.get("/models/{model_id}", tags=["models"])
def get_model(model_id: int, svc: ModelsMySQLService = Depends(get_models_service)):
    rows = svc.retrieve(model_id=model_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Model not found")
    return rows[0]

@app.post("/models", tags=["models"], status_code=status.HTTP_201_CREATED)
def register_model(model_input: ModelInput, svc: ModelsMySQLService = Depends(get_models_service)):
    payload = {
        "name": model_input.name,
        "version": model_input.version,
        "model_type": model_input.model_type,
        "artifact_uri": getattr(model_input, "model_path", None) or "/models/unknown.bin",
        "input_shape": model_input.input_shape,
        "output_shape": [model_input.output_classes],
    }
    mid = svc.create(payload)
    return {"model_id": mid}

@app.delete("/models/{model_id}", tags=["models"])
def delete_model(model_id: int, svc: ModelsMySQLService = Depends(get_models_service)):
    ok = svc.delete(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"deleted": True}


# -----------------------------------------------------------------------------
# Prediction Batch Routes
# # -----------------------------------------------------------------------------

# @app.get("/predictions", response_model=List[PredictionRequest], tags=["predictions"])
# async def get_all_predictions():
#     """
#     Retrieve all batch prediction requests.
#     """
#     if not predictions_db:
#         # Return some dummy data for demonstration
#         dummy_prediction = make_dummy_prediction()
#         predictions_db[dummy_prediction.id] = dummy_prediction
    
#     return list(predictions_db.values())

# @app.get("/predictions/{prediction_id}", response_model=PredictionRequest, tags=["predictions"])
# async def get_prediction(prediction_id: UUID):
#     """
#     Retrieve a specific batch prediction request by ID.
#     """
#     if prediction_id not in predictions_db:
#         # Create dummy data if not found
#         dummy_prediction = make_dummy_prediction(prediction_id)
#         predictions_db[prediction_id] = dummy_prediction
    
#     return predictions_db[prediction_id]

# @app.post("/predictions", response_model=PredictionRequest, status_code=status.HTTP_201_CREATED, tags=["predictions"])
# async def create_batch_prediction(prediction_request: PredictionRequest):
#     """
#     Create a new batch prediction request (NOT IMPLEMENTED).
#     """
#     raise HTTPException(
#         status_code=status.HTTP_501_NOT_IMPLEMENTED,
#         detail="NOT IMPLEMENTED: Batch prediction processing not yet available"
#     )

# @app.put("/predictions/{prediction_id}", response_model=PredictionRequest, tags=["predictions"])
# async def update_prediction(prediction_id: UUID, prediction_request: PredictionRequest):
#     """
#     Update a batch prediction request (NOT IMPLEMENTED).
#     """
#     raise HTTPException(
#         status_code=status.HTTP_501_NOT_IMPLEMENTED,
#         detail="NOT IMPLEMENTED: Prediction update functionality not yet available"
#     )

# @app.delete("/predictions/{prediction_id}", tags=["predictions"])
# async def delete_prediction(prediction_id: UUID):
#     """
#     Delete a batch prediction request.
#     """
#     if prediction_id in predictions_db:
#         del predictions_db[prediction_id]
#         return {"message": f"Prediction request {prediction_id} deleted successfully"}
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Prediction request with ID {prediction_id} not found"
#         )

# List
@app.get("/predictions", tags=["predictions"])
def list_predictions(
    session_id: int | None = None,
    limit: int = 100,
    svc: PredictionsMySQLService = Depends(get_predictions_service),
):
    return svc.retrieve(session_id=session_id, limit=limit)

# Get one
@app.get("/predictions/{prediction_id}", tags=["predictions"])
def get_prediction(prediction_id: int, svc: PredictionsMySQLService = Depends(get_predictions_service)):
    row = svc.get_one(prediction_id)
    if not row:
        raise HTTPException(404, "Prediction not found")
    # If params is JSON text, parse it for nicer API output
    if row.get("params"):
        try:
            row["params"] = json.loads(row["params"])
        except Exception:
            pass
    return row

# Create
@app.post("/predictions", tags=["predictions"], status_code=status.HTTP_201_CREATED)
def create_prediction(
    payload: dict = Body(...),  # or a dedicated Pydantic model if you want stricter validation
    svc: PredictionsMySQLService = Depends(get_predictions_service),
):
    pid = svc.create({
        "requestor_user_id": payload.get("requestor_user_id"),
        "session_id": payload.get("session_id"),
        "model_id": payload.get("model_id"),
        "status": payload.get("status", "queued"),
        "params": payload.get("params", {}),
    })
    return {"prediction_id": pid, "status": "queued"}

# Update (status or completion)
@app.put("/predictions/{prediction_id}", tags=["predictions"])
def update_prediction(
    prediction_id: int,
    payload: dict = Body(...),  # e.g., {"status":"running"} OR {"output_text":"...", "confidence":0.9}
    svc: PredictionsMySQLService = Depends(get_predictions_service),
):
    ok = svc.update(prediction_id, payload)
    if not ok:
        raise HTTPException(404, "Prediction not found or nothing to update")
    return {"updated": True}

# Delete
@app.delete("/predictions/{prediction_id}", tags=["predictions"])
def delete_prediction(prediction_id: int, svc: PredictionsMySQLService = Depends(get_predictions_service)):
    ok = svc.delete(prediction_id)
    if not ok:
        raise HTTPException(404, "Prediction not found")
    return {"deleted": True}


# -----------------------------------------------------------------------------
# Health and Info Routes
# -----------------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root():
    """
    Root endpoint providing basic service information.
    """
    return {
        "service": "Model Serving Microservice",
        "version": "1.0.0",
        "description": "A microservice for serving machine learning models for hand gesture recognition",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "gestures": "/gestures",
            "models": "/models",
            "predictions": "/predictions"
        }
    }

# @app.get("/health", tags=["health"])
# async def health_check():
#     """
#     Health check endpoint for monitoring and load balancing.
#     """
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "service": "Model Serving Microservice",
#         "version": "1.0.0",
#         "database_status": "connected",  # In production, check actual DB connection
#         "model_status": "loaded"  # In production, check if models are loaded
#     }

@app.get("/health", tags=["health"])
async def health_check():
    db_ok = False
    try:
        svc = ModelsMySQLService()
        conn = svc.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        cur.close()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Model Serving Microservice",
        "version": "1.0.0",
        "database_status": "connected" if db_ok else "unreachable",
        "model_status": "loaded"
    }


# -----------------------------------------------------------------------------
# Run the application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)