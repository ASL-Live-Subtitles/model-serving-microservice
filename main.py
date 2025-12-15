import io
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import aiomysql
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Header, Depends
from PIL import Image
from pydantic import BaseModel, Field

# Import our models
from models.landmark import LandmarkInput, LandmarkResult
from models.gloss import GlossPredictionInput, GlossPredictionResult

# Initialize FastAPI app with OpenAPI documentation
app = FastAPI(
    title="Model Serving Microservice",
    description="Model Serving microservice for extracting hand landmarks and predicting ASL gloss tokens",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "landmarks",
            "description": "Extract hand landmarks from images/frames (stubbed MediaPipe)"
        },
        {
            "name": "gloss-predictions",
            "description": "Predict ASL gloss tokens from landmarks"
        },
        {
            "name": "video-gloss",
            "description": "Submit a recorded video and get a single gloss + dubbed sentence"
        }
    ]
)

try:
    import mediapipe as mp

    _mp_hands = mp.solutions.hands
    HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover - optional dependency
    mp = None
    _mp_hands = None
    HAS_MEDIAPIPE = False
try:
    from openai import OpenAI

    HAS_OPENAI = True
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    HAS_OPENAI = False

# In-memory storage for demonstration (replace with actual database in production)
landmarks_db: Dict[UUID, LandmarkResult] = {}
gloss_db: Dict[UUID, GlossPredictionResult] = {}

# Dummy helper functions
def make_dummy_landmarks(landmark_id: UUID = None) -> LandmarkResult:
    """Create a dummy landmark extraction result."""
    return LandmarkResult(
        id=landmark_id or uuid4(),
        session_id="session-123",
        user_id="demo_user",
        landmarks=[[0.5, 0.6], [0.52, 0.58], [0.54, 0.56], [0.56, 0.54], [0.58, 0.52],
                   [0.48, 0.62], [0.46, 0.64], [0.44, 0.66], [0.42, 0.68], [0.40, 0.70],
                   [0.60, 0.50], [0.62, 0.48], [0.64, 0.46], [0.66, 0.44], [0.68, 0.42],
                   [0.38, 0.72], [0.36, 0.74], [0.34, 0.76], [0.32, 0.78], [0.30, 0.80],
                   [0.70, 0.40]],
        method="stub",
        extracted_at=datetime.utcnow(),
        metadata={"device": "demo"}
    )

def make_dummy_gloss(gloss_id: UUID = None, landmark_id: UUID = None) -> GlossPredictionResult:
    """Create a dummy gloss prediction result."""
    return GlossPredictionResult(
        id=gloss_id or uuid4(),
        session_id="session-123",
        user_id="demo_user",
        landmark_id=landmark_id or uuid4(),
        gloss=["HELLO"],
        confidence=0.93,
        inference_method="stub",
        model_version="v1.0.0",
        created_at=datetime.utcnow()
    )

class VideoGlossResponse(BaseModel):
    """Response for video-based gloss prediction."""

    gloss: List[str] = Field(..., description="Predicted ASL gloss tokens (prefix removed)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    frames_processed: int = Field(..., description="Number of frames used")
    inference_method: str = Field(..., description="Method used (openai/heuristic/stub)")
    model_version: str = Field("v1.0.0", description="Model version used")
    raw_openai_response: Optional[dict] = Field(
        None,
        description="Raw OpenAI response payload (if OpenAI was used)",
    )


def decode_image_file(upload: UploadFile) -> np.ndarray:
    """Load an uploaded image into a numpy RGB array."""
    try:
        data = upload.file.read()
        if not data:
            raise ValueError("Empty file content.")
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image)
    except Exception as exc:  # pragma: no cover - error path
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read image: {exc}"
        ) from exc


def extract_landmarks_from_image(image_np: np.ndarray) -> (List[List[float]], str):
    """Run MediaPipe Hands on an RGB image array; fallback to stub if unavailable."""
    if HAS_MEDIAPIPE:
        with _mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            results = hands.process(image_np)
            if results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                return coords, "mediapipe"
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No hand detected in image."
            )
    # Fallback: stub landmarks
    stub = make_dummy_landmarks()
    return stub.landmarks, "stub"


def extract_landmarks_from_video(video_bytes: bytes, target_fps: int = 20) -> Tuple[List[List[List[float]]], str, bool]:
    """
    Extract landmarks from a video by sampling frames at target_fps.
    Returns list of landmark frames, method used, and whether any hands were detected.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to read video."
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(fps / target_fps)))
        frame_idx = 0
        all_landmarks: List[List[List[float]]] = []
        method_used = "stub"
        detected_any = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                coords, method = extract_landmarks_from_image(rgb)
                method_used = method
                detected_any = True
                all_landmarks.append(coords)
            except HTTPException:
                # Skip frames without detected hands; continue sampling.
                pass
            frame_idx += 1
        cap.release()
    if not all_landmarks:
        # Fallback to a single stub frame so the pipeline continues for demo purposes.
        stub = make_dummy_landmarks()
        return [stub.landmarks], "stub", False
    return all_landmarks, method_used, detected_any


def _normalize_gloss_to_list(gloss_raw: str) -> List[str]:
    cleaned = gloss_raw.replace("GLOSS_", "")
    tokens = [token for token in cleaned.replace("-", "_").replace(" ", "_").split("_") if token]
    return [t.upper() for t in tokens] if tokens else ["DEFAULT"]


def infer_gloss_from_landmarks(landmarks: List[List[float]], api_key: Optional[str] = None) -> Tuple[List[str], float, str, Optional[dict]]:
    """Predict gloss from landmarks using OpenAI if available; fallback to dubbed gloss."""
    arr = np.array(landmarks, dtype=float)
    if arr.shape != (21, 2):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="landmarks must be a 21x2 array."
        )
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if HAS_OPENAI and resolved_key:
        try:
            client = OpenAI(api_key=resolved_key)
            prompt = (
                "You are an ASL gloss predictor. "
                "Given 21 normalized hand landmark (x,y) pairs, output a single ASL gloss token "
                "such as A-Z letter, HELLO, THANK-YOU, YES, NO, etc. Use uppercase with underscores, no extra words.\n"
                f"Landmarks: {arr.tolist()}"
            )
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=6,
                temperature=0.2,
            )
            gloss_raw = resp.choices[0].message.content.strip()
            gloss_tokens = _normalize_gloss_to_list(gloss_raw)
            raw_payload = resp.model_dump()
            return gloss_tokens, 0.9, "openai", raw_payload
        except Exception:
            # Fall through to fallback gloss below.
            pass

    # Fallback dubbed gloss when OpenAI is unavailable or fails.
    return ["FINALLY", "CLASS", "LAST", "PRESENTATION"], 0.5, "fallback", None


def infer_gloss_from_sequence(
    landmark_sequence: List[List[List[float]]],
    api_key: Optional[str] = None
) -> Tuple[List[str], float, str, Optional[dict]]:
    """Aggregate a sequence of landmarks into one gloss prediction."""
    seq_arr = np.array(landmark_sequence, dtype=float)
    if seq_arr.ndim != 3 or seq_arr.shape[1:] != (21, 2):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="landmark_sequence must have shape (T, 21, 2)."
        )
    # Simple aggregation: mean across time, then reuse gloss inference
    mean_landmarks = seq_arr.mean(axis=0).tolist()
    gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(mean_landmarks, api_key=api_key)
    return gloss, confidence, method, raw_payload


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

async def get_pool():
    """
    Lazily create a global aiomysql pool using environment variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SOCKET (optional).
    """
    if not hasattr(app.state, "db_pool"):
        app.state.db_pool = await aiomysql.create_pool(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME", "model_serving_db"),
            unix_socket=os.getenv("DB_SOCKET") or None,
            autocommit=True,
        )
    return app.state.db_pool


async def save_video_gloss_request(
    pool,
    req_id: UUID,
    video_uri: Optional[str],
    landmarks: List[List[List[float]]],
    gloss: List[str],
    confidence: float,
    method: str,
    raw_openai: Optional[dict],
    detected: bool,
):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO video_gloss_requests
                (id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(req_id),
                    video_uri,
                    json.dumps(landmarks),
                    json.dumps(gloss),
                    confidence,
                    method,
                    json.dumps(raw_openai) if raw_openai else None,
                    detected,
                ),
        )


async def delete_video_gloss_request(pool, req_id: UUID):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM video_gloss_requests WHERE id = %s",
                (str(req_id),),
            )

@app.on_event("shutdown")
async def close_pool():
    pool = getattr(app.state, "db_pool", None)
    if pool:
        pool.close()
        await pool.wait_closed()

# -----------------------------------------------------------------------------
# Landmark Routes (image/frame -> landmarks)
# -----------------------------------------------------------------------------

@app.post("/landmarks", response_model=LandmarkResult, status_code=status.HTTP_201_CREATED, tags=["landmarks"])
async def create_landmarks(
    file: UploadFile = File(..., description="Image or frame containing a hand"),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    frame_index: Optional[int] = Form(None)
):
    """
    Extract 21 hand landmarks from an image or frame (stubbed MediaPipe).
    """
    image_np = decode_image_file(file)
    coords, method = extract_landmarks_from_image(image_np)

    result = make_dummy_landmarks()
    result.landmarks = coords
    result.method = method
    result.session_id = session_id
    result.user_id = user_id
    result.metadata = {"frame_index": str(frame_index)} if frame_index is not None else None
    landmarks_db[result.id] = result
    return result

@app.get("/landmarks", response_model=List[LandmarkResult], tags=["landmarks"])
async def list_landmarks():
    """
    List all landmark extraction results (stubbed).
    """
    if not landmarks_db:
        dummy = make_dummy_landmarks()
        landmarks_db[dummy.id] = dummy
    return list(landmarks_db.values())

@app.get("/landmarks/{landmark_id}", response_model=LandmarkResult, tags=["landmarks"])
async def get_landmarks(landmark_id: UUID):
    """
    Retrieve a specific landmark extraction result.
    """
    if landmark_id not in landmarks_db:
        landmarks_db[landmark_id] = make_dummy_landmarks(landmark_id)
    return landmarks_db[landmark_id]

@app.put("/landmarks/{landmark_id}", response_model=LandmarkResult, tags=["landmarks"])
async def update_landmarks(landmark_id: UUID, request: LandmarkInput):
    """
    Update a landmark record (NOT IMPLEMENTED).
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="NOT IMPLEMENTED: Landmark update functionality not yet available"
    )

@app.delete("/landmarks/{landmark_id}", tags=["landmarks"])
async def delete_landmarks(landmark_id: UUID):
    """
    Delete a landmark record.
    """
    if landmark_id in landmarks_db:
        del landmarks_db[landmark_id]
        return {"message": f"Landmark {landmark_id} deleted successfully"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Landmark with ID {landmark_id} not found"
    )

# -----------------------------------------------------------------------------
# Gloss Prediction Routes (landmarks -> gloss)
# -----------------------------------------------------------------------------

@app.post("/gloss-predictions", response_model=GlossPredictionResult, status_code=status.HTTP_201_CREATED, tags=["gloss-predictions"])
async def create_gloss_prediction(
    request: GlossPredictionInput,
    openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    pool=Depends(get_pool),
):
    """
    Predict ASL gloss tokens from 21-point landmarks (stubbed).
    """
    if not request.landmark_id and not request.landmarks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either landmark_id or landmarks."
        )

    if request.landmarks:
        if len(request.landmarks) != 21:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="landmarks must contain exactly 21 points."
            )

    landmark_id = request.landmark_id
    if landmark_id and landmark_id not in landmarks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Landmark with ID {landmark_id} not found"
        )

    result = make_dummy_gloss(landmark_id=landmark_id)
    # If inline landmarks are provided, infer gloss directly from them and store.
    raw_payload = None
    if request.landmarks:
        gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(request.landmarks, api_key=openai_key)
        result.gloss = gloss
        result.confidence = confidence
        result.inference_method = method
    elif landmark_id:
        landmarks = landmarks_db[landmark_id].landmarks
        gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(landmarks, api_key=openai_key)
        result.gloss = gloss
        result.confidence = confidence
        result.inference_method = method
    result.session_id = request.session_id
    result.user_id = request.user_id
    result.landmark_id = landmark_id or result.landmark_id
    result.raw_openai_response = raw_payload
    if request.landmarks:
        # Optionally cache inline landmarks as a new record for traceability.
        new_landmark = make_dummy_landmarks()
        new_landmark.landmarks = request.landmarks
        new_landmark.session_id = request.session_id
        new_landmark.user_id = request.user_id
        landmarks_db[new_landmark.id] = new_landmark
        result.landmark_id = new_landmark.id
    gloss_db[result.id] = result

    # Persist to DB (uses provided landmarks or referenced landmark)
    landmarks_payload = request.landmarks or landmarks_db.get(result.landmark_id, make_dummy_landmarks()).landmarks
    try:
        await save_video_gloss_request(
            pool=pool,
            req_id=result.id,
            video_uri=None,
            landmarks=[landmarks_payload],
            gloss=result.gloss,
            confidence=result.confidence,
            method=result.inference_method,
            raw_openai=result.raw_openai_response,
            detected=True,
        )
    except Exception:
        # Log/ignore DB errors to keep API responsive
        pass

    return result

@app.get("/gloss-predictions", response_model=List[GlossPredictionResult], tags=["gloss-predictions"])
async def list_gloss_predictions():
    """
    List gloss prediction results (stubbed).
    """
    if not gloss_db:
        dummy = make_dummy_gloss()
        gloss_db[dummy.id] = dummy
    return list(gloss_db.values())

@app.get("/gloss-predictions/{gloss_id}", response_model=GlossPredictionResult, tags=["gloss-predictions"])
async def get_gloss_prediction(gloss_id: UUID):
    """
    Retrieve a specific gloss prediction result.
    """
    if gloss_id not in gloss_db:
        gloss_db[gloss_id] = make_dummy_gloss(gloss_id)
    return gloss_db[gloss_id]

@app.put("/gloss-predictions/{gloss_id}", response_model=GlossPredictionResult, tags=["gloss-predictions"])
async def update_gloss_prediction(
    gloss_id: UUID,
    request: GlossPredictionInput,
    openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    pool=Depends(get_pool),
):
    """
    Update a gloss prediction by recomputing it from provided landmarks or landmark_id.
    """
    if gloss_id not in gloss_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gloss prediction with ID {gloss_id} not found"
        )
    if not request.landmark_id and not request.landmarks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either landmark_id or landmarks."
        )
    if request.landmarks and len(request.landmarks) != 21:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="landmarks must contain exactly 21 points."
        )

    landmark_id = request.landmark_id
    if landmark_id and landmark_id not in landmarks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Landmark with ID {landmark_id} not found"
        )

    result = gloss_db[gloss_id]
    # Recompute gloss
    raw_payload = None
    if request.landmarks:
        gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(request.landmarks, api_key=openai_key)
        # Cache new landmarks for traceability
        new_landmark = make_dummy_landmarks()
        new_landmark.landmarks = request.landmarks
        new_landmark.session_id = request.session_id
        new_landmark.user_id = request.user_id
        landmarks_db[new_landmark.id] = new_landmark
        result.landmark_id = new_landmark.id
    else:
        landmarks = landmarks_db[landmark_id].landmarks
        gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(landmarks, api_key=openai_key)
        result.landmark_id = landmark_id

    result.gloss = gloss
    result.confidence = confidence
    result.inference_method = method
    result.session_id = request.session_id
    result.user_id = request.user_id
    result.raw_openai_response = raw_payload
    gloss_db[gloss_id] = result

    landmarks_payload = request.landmarks or landmarks_db.get(result.landmark_id, make_dummy_landmarks()).landmarks
    try:
        await save_video_gloss_request(
            pool=pool,
            req_id=result.id,
            video_uri=None,
            landmarks=[landmarks_payload],
            gloss=result.gloss,
            confidence=result.confidence,
            method=result.inference_method,
            raw_openai=result.raw_openai_response,
            detected=True,
        )
    except Exception:
        pass

    return result

@app.delete("/gloss-predictions/{gloss_id}", tags=["gloss-predictions"])
async def delete_gloss_prediction(gloss_id: UUID, pool=Depends(get_pool)):
    """
    Delete a gloss prediction record.
    """
    if gloss_id in gloss_db:
        del gloss_db[gloss_id]
        try:
            await delete_video_gloss_request(pool, gloss_id)
        except Exception:
            pass
        return {"message": f"Gloss prediction {gloss_id} deleted successfully"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Gloss prediction with ID {gloss_id} not found"
    )

# -----------------------------------------------------------------------------
# Video Gloss Route (video -> sequence -> gloss + sentence)
# -----------------------------------------------------------------------------

@app.post("/video-gloss", response_model=VideoGlossResponse, status_code=status.HTTP_201_CREATED, tags=["video-gloss"])
async def create_video_gloss(
    file: UploadFile = File(..., description="Recorded ASL video clip"),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    target_fps: int = Form(5),
    openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    pool=Depends(get_pool),
):
    """
    Submit a recorded video, extract landmarks per frame, run temporal inference, and return one gloss + dubbed sentence.
    """
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty video file."
        )
    landmarks_seq, method_used, detected_any = extract_landmarks_from_video(video_bytes, target_fps=target_fps)

    raw_payload = None
    if not detected_any:
        # Force a demo-friendly fallback gloss/sentence when no hand is detected.
        gloss = ["FINALLY", "CLASS", "LAST", "PRESENTATION"]
        confidence = 0.5
        method = "fallback"
    else:
        gloss, confidence, method, raw_payload = infer_gloss_from_sequence(landmarks_seq, api_key=openai_key)

    response = VideoGlossResponse(
        gloss=gloss,
        confidence=confidence,
        frames_processed=len(landmarks_seq),
        inference_method=method if method != "heuristic" else method_used,
        model_version="v1.0.0",
        raw_openai_response=raw_payload,
    )

    try:
        await save_video_gloss_request(
            pool=pool,
            req_id=uuid4(),
            video_uri=file.filename,
            landmarks=landmarks_seq,
            gloss=gloss,
            confidence=confidence,
            method=response.inference_method,
            raw_openai=raw_payload,
            detected=detected_any,
        )
    except Exception:
        pass

    return response

# -----------------------------------------------------------------------------
# Health and Info Routes
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """
    Root endpoint providing basic service information.
    """
    return {
        "service": "Model Serving Microservice",
        "version": "1.0.0",
        "description": "Extract hand landmarks from images/frames and predict ASL gloss tokens",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "landmarks": "/landmarks",
            "gloss_predictions": "/gloss-predictions",
        },
    }

# -----------------------------------------------------------------------------
# Run the application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
