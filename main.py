import base64
import io
import math
import os
import json
import re
import tempfile
import random
import traceback
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
from models.landmark import LandmarkInput, LandmarkResult, LandmarkUpdate
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
except Exception:
    OpenAI = None
    HAS_OPENAI = False

# Cache OpenAI client
_openai_client = None

# In-memory storage for testing (used when DB is unavailable)
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


def extract_frame_base64(video_bytes: bytes) -> Optional[str]:
    """Grab a representative frame from the video and return as base64-encoded JPEG."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_idx = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, buf = cv2.imencode(".jpg", rgb)
        if not success:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")

def infer_gloss_from_landmarks(
    landmarks: List[List[float]],
    api_key: Optional[str] = None
) -> Tuple[List[str], float, str, Optional[dict]]:
    """
    Predict ASL gloss from 21x2 hand landmarks using OpenAI if available.
    Falls back to a random dummy gloss otherwise.
    """

    arr = np.array(landmarks, dtype=float)
    if arr.shape != (21, 2):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="landmarks must be a 21x2 array."
        )

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    dummy_response = ["HELLO", "THANK_YOU", "GOODBYE"]

    if resolved_key and OpenAI is not None and callable(OpenAI):
        try:
            print("[gloss] starting OpenAI inference")
            global _openai_client
            if _openai_client is None:
                _openai_client = OpenAI(api_key=resolved_key)
            client = _openai_client

            prompt = (
                "You are an ASL gloss predictor.\n"
                "Given 21 normalized hand landmark (x,y) pairs, output JSON:\n"
                '{ "gloss": "<ASL_GLOSS>", "confidence": <0-1> }\n'
                "Rules:\n"
                "- gloss must be uppercase with underscores and no spaces\n"
                "- confidence must be a float between 0 and 1\n"
                "- no extra text or explanation\n"
                "- respond with complete JSON on one line\n\n"
                f"Landmarks:\n{arr.tolist()}"
            )

            resp = client.responses.create(
                model="gpt-4o",
                input=prompt,
                max_output_tokens=50,
                temperature=0.2,
            )

            gloss_raw = resp.output_text.strip()
            print(f"[gloss] raw OpenAI text: {gloss_raw!r}")
            try:
                # Handle fenced code blocks like ```json ... ```
                payload = gloss_raw
                if payload.startswith("```"):
                    payload = payload.strip("`")
                    if payload.lower().startswith("json"):
                        payload = payload[4:]
                data = json.loads(payload)
                gloss_token = data.get("gloss")
                confidence = float(data.get("confidence", 0.0))
                if not gloss_token:
                    raise ValueError("Missing gloss")
                gloss_tokens = [gloss_token]
            except Exception:
                # Regex fallback if JSON is truncated (e.g., missing closing brace)
                gloss_match = re.search(r'"gloss"\\s*:\\s*"([^"]+)"', gloss_raw)
                conf_match = re.search(r'"confidence"\\s*:\\s*([0-9]*\\.?[0-9]+)', gloss_raw)
                gloss_token = gloss_match.group(1) if gloss_match else gloss_raw
                confidence = float(conf_match.group(1)) if conf_match else 0.5
                gloss_tokens = [gloss_token]
                print("[gloss] Parsed via regex fallback")

            print(f"[gloss] tokens={gloss_tokens}, confidence={confidence}")

            return gloss_tokens, confidence, "openai", resp.model_dump()

        except Exception as e:
            print("[gloss] OpenAI inference failed:", repr(e))
            traceback.print_exc()
    else:
        print("[gloss] OpenAI client class not available; using fallback gloss.")

    fallback_gloss = random.choice(dummy_response)
    print(f"[gloss] using fallback gloss: {fallback_gloss}")
    return [fallback_gloss], 0.0, "fallback", None



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

async def get_pool():
    """
    Lazily create a global aiomysql pool using environment variables:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SOCKET (optional).
    """
    if not hasattr(app.state, "db_pool"):
        try:
            app.state.db_pool = await aiomysql.create_pool(
                host=os.getenv("DB_HOST", "127.0.0.1"),
                port=int(os.getenv("DB_PORT", "3306")),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                db=os.getenv("DB_NAME", "model_serving_db"),
                unix_socket=os.getenv("DB_SOCKET") or None,
                autocommit=True,
            )
            print("[db] connection pool created")
        except Exception as e:
            # Fallback for development if DB is not available
            print(f"Warning: Database connection failed ({e}). Running without DB.")
            return None
    else:
        if app.state.db_pool:
            print("[db] connection pool reused")
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
    if not pool:
        return
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


def _gloss_row_to_result(row) -> GlossPredictionResult:
    """
    Map a DB row from video_gloss_requests to GlossPredictionResult.
    Expects columns: id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at (optional).
    """
    (
        id_,
        _video_uri,
        landmarks_json,
        gloss_json,
        confidence,
        inference_method,
        raw_openai_json,
        _detected,
        *rest,
    ) = row

    created_at = rest[0] if rest else datetime.utcnow()
    gloss_list = json.loads(gloss_json) if isinstance(gloss_json, (str, bytes)) else gloss_json
    raw_openai = json.loads(raw_openai_json) if isinstance(raw_openai_json, (str, bytes)) else raw_openai_json

    return GlossPredictionResult(
        id=UUID(id_),
        session_id=None,
        user_id=None,
        landmark_id=None,
        gloss=gloss_list,
        confidence=float(confidence) if confidence is not None else 0.0,
        inference_method=inference_method or "fallback",
        model_version="v1.0.0",
        raw_openai_response=raw_openai,
        created_at=created_at,
    )


def _video_gloss_row_to_response(row) -> VideoGlossResponse:
    """
    Map a DB row from video_gloss_requests to VideoGlossResponse.
    Expects columns: id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at (optional).
    """
    if len(row) < 8:
        raise ValueError(f"Unexpected row length: {len(row)}")

    id_ = row[0]
    _video_uri = row[1]
    landmarks_json = row[2]
    gloss_json = row[3]
    confidence = row[4]
    inference_method = row[5]
    raw_openai_json = row[6]
    detected = row[7] if len(row) > 7 else None
    created_at = row[8] if len(row) > 8 else datetime.utcnow()

    def _safe_json(val, default):
        if val is None:
            return default
        if isinstance(val, (str, bytes)):
            try:
                return json.loads(val)
            except Exception:
                return default
        return val

    landmarks_list = _safe_json(landmarks_json, [])
    gloss_list = _safe_json(gloss_json, [])
    raw_openai = _safe_json(raw_openai_json, None)

    frames_processed = len(landmarks_list) if isinstance(landmarks_list, list) else 0

    return VideoGlossResponse(
        id=str(id_),
        gloss=gloss_list or ["DEFAULT"],
        confidence=float(confidence) if confidence is not None else 0.0,
        frames_processed=frames_processed,
        inference_method=inference_method or "fallback",
        model_version="v1.0.0",
        raw_openai_response=raw_openai,
    )


async def fetch_video_gloss_by_id(pool, gloss_id: UUID):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at
                FROM video_gloss_requests
                WHERE id = %s
                """,
                (str(gloss_id),),
            )
            return await cur.fetchone()

@app.on_event("shutdown")
async def close_pool():
    pool = getattr(app.state, "db_pool", None)
    if pool:
        pool.close()
        await pool.wait_closed()

# Landmark Routes
@app.post("/landmarks", response_model=LandmarkResult, status_code=status.HTTP_201_CREATED, tags=["landmarks"])
async def create_landmarks(
    file: UploadFile = File(..., description="Image or frame containing a hand"),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    frame_index: Optional[int] = Form(None),
    pool=Depends(get_pool),
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
    return list(landmarks_db.values())

@app.get("/landmarks/{landmark_id}", response_model=LandmarkResult, tags=["landmarks"])
async def get_landmarks(landmark_id: UUID):
    """
    Retrieve a specific landmark extraction result.
    """
    if landmark_id not in landmarks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Landmark with ID {landmark_id} not found"
        )
    return landmarks_db[landmark_id]

@app.put("/landmarks/{landmark_id}", response_model=LandmarkResult, tags=["landmarks"])
async def update_landmarks(
    landmark_id: UUID,
    request: LandmarkUpdate,
    pool=Depends(get_pool),
):
    """
    Update a landmark record (in-memory + best-effort DB persist).
    """
    record = landmarks_db.get(landmark_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Landmark with ID {landmark_id} not found"
        )
    if request.landmarks and len(request.landmarks) != 21:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="landmarks must contain exactly 21 points."
        )

    if request.landmarks:
        record.landmarks = request.landmarks
    if request.session_id is not None:
        record.session_id = request.session_id
    if request.user_id is not None:
        record.user_id = request.user_id
    if request.metadata is not None:
        record.metadata = request.metadata

    landmarks_db[landmark_id] = record
    return record

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

# Gloss Prediction Routes
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

    source_landmarks = request.landmarks or (landmarks_db[landmark_id].landmarks if landmark_id else None)
    if not source_landmarks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="landmarks not found for provided landmark_id"
        )
    gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(source_landmarks, api_key=openai_key)

    result = GlossPredictionResult(
        id=uuid4(),
        session_id=request.session_id,
        user_id=request.user_id,
        landmark_id=landmark_id,
        gloss=gloss,
        confidence=confidence,
        inference_method=method,
        model_version="v1.0.0",
        raw_openai_response=raw_payload,
        created_at=datetime.utcnow(),
    )

    if request.landmarks:
        cached_landmark = make_dummy_landmarks()
        cached_landmark.landmarks = request.landmarks
        cached_landmark.session_id = request.session_id
        cached_landmark.user_id = request.user_id
        landmarks_db[cached_landmark.id] = cached_landmark
        result.landmark_id = cached_landmark.id
    else:
        result.landmark_id = landmark_id

    if not pool:
        gloss_db[result.id] = result

    # Persist to DB (uses provided landmarks or referenced landmark)
    landmarks_payload = request.landmarks or landmarks_db[result.landmark_id].landmarks
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
        print("Warning: Failed to save gloss prediction to DB.")
        pass

    return result

@app.get("/gloss-predictions", response_model=List[GlossPredictionResult], tags=["gloss-predictions"])
async def list_gloss_predictions():
    """
    List gloss prediction results (stubbed).
    """
    pool = await get_pool()
    if pool:
        results: List[GlossPredictionResult] = []
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at FROM video_gloss_requests"
                )
                rows = await cur.fetchall()
                for row in rows:
                    try:
                        results.append(_gloss_row_to_result(row))
                    except Exception:
                        continue
        return results

    if not gloss_db:
        dummy = make_dummy_gloss()
        gloss_db[dummy.id] = dummy
    return list(gloss_db.values())

@app.get("/gloss-predictions/{gloss_id}", response_model=GlossPredictionResult, tags=["gloss-predictions"])
async def get_gloss_prediction(gloss_id: UUID):
    """
    Retrieve a specific gloss prediction result.
    """
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at
                    FROM video_gloss_requests
                    WHERE id = %s
                    """,
                    (str(gloss_id),),
                )
                row = await cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Gloss prediction with ID {gloss_id} not found"
                    )
                return _gloss_row_to_result(row)

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
    existing = None
    if pool:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at
                    FROM video_gloss_requests
                    WHERE id = %s
                    """,
                    (str(gloss_id),),
                )
                row = await cur.fetchone()
                if row:
                    existing = _gloss_row_to_result(row)
    if not existing and gloss_id in gloss_db:
        existing = gloss_db[gloss_id]
    if not existing:
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
    landmark_record = None
    if landmark_id:
        if landmark_id in landmarks_db:
            landmark_record = landmarks_db[landmark_id]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Landmark with ID {landmark_id} not found"
            )

    if request.landmarks:
        source_landmarks = request.landmarks
    elif landmark_record:
        source_landmarks = landmark_record.landmarks
    else:
        stored_landmarks = None
        if pool:
            try:
                async with pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT landmarks FROM video_gloss_requests WHERE id = %s",
                            (str(gloss_id),),
                        )
                        row = await cur.fetchone()
                        if row and row[0]:
                            parsed = json.loads(row[0]) if isinstance(row[0], (str, bytes)) else row[0]
                            if isinstance(parsed, list) and parsed:
                                candidate = parsed[0] if isinstance(parsed[0], list) else parsed
                                if isinstance(candidate, list) and len(candidate) == 21:
                                    stored_landmarks = candidate
                            elif isinstance(parsed, list) and len(parsed) == 21:
                                stored_landmarks = parsed
            except Exception:
                stored_landmarks = None

        if stored_landmarks is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="landmarks not found for provided landmark_id or existing record"
            )
        source_landmarks = stored_landmarks

    gloss, confidence, method, raw_payload = infer_gloss_from_landmarks(source_landmarks, api_key=openai_key)

    result = existing
    if request.landmarks:
        new_landmark = make_dummy_landmarks()
        new_landmark.landmarks = request.landmarks
        new_landmark.session_id = request.session_id
        new_landmark.user_id = request.user_id
        landmarks_db[new_landmark.id] = new_landmark
        result.landmark_id = new_landmark.id
    else:
        result.landmark_id = landmark_id

    result.gloss = gloss
    result.confidence = confidence
    result.inference_method = method
    result.session_id = request.session_id
    result.user_id = request.user_id
    result.raw_openai_response = raw_payload
    if not pool:
        gloss_db[gloss_id] = result

    landmarks_payload = request.landmarks or (
        landmark_record.landmarks if landmark_record else landmarks_db.get(result.landmark_id, make_dummy_landmarks()).landmarks
    )
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
    if pool:
        try:
            await delete_video_gloss_request(pool, gloss_id)
            return {"message": f"Gloss prediction {gloss_id} deleted successfully"}
        except Exception:
            pass

    if gloss_id in gloss_db:
        del gloss_db[gloss_id]
        return {"message": f"Gloss prediction {gloss_id} deleted successfully"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Gloss prediction with ID {gloss_id} not found"
    )

# Video Gloss Routes (DB-backed)
@app.get("/video-gloss", response_model=List[VideoGlossResponse], tags=["video-gloss"])
async def list_video_gloss(pool=Depends(get_pool)):
    if not pool:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    results: List[VideoGlossResponse] = []
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id, video_uri, landmarks, gloss, confidence, inference_method, raw_openai_response, detected, created_at FROM video_gloss_requests"
            )
            rows = await cur.fetchall()
            print(f"[video-gloss] fetched {len(rows)} rows from DB")
            for row in rows:
                try:
                    results.append(_video_gloss_row_to_response(row))
                except Exception:
                    print("[video-gloss] failed to parse a DB row")
                    continue
    return results


@app.get("/video-gloss/{gloss_id}", response_model=VideoGlossResponse, tags=["video-gloss"])
async def get_video_gloss(gloss_id: UUID, pool=Depends(get_pool)):
    if not pool:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    row = await fetch_video_gloss_by_id(pool, gloss_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video gloss with ID {gloss_id} not found")
    return _video_gloss_row_to_response(row)


@app.put("/video-gloss/{gloss_id}", response_model=VideoGlossResponse, tags=["video-gloss"])
async def update_video_gloss(gloss_id: UUID, payload: VideoGlossUpdate, pool=Depends(get_pool)):
    if not pool:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    existing_row = await fetch_video_gloss_by_id(pool, gloss_id)
    if not existing_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video gloss with ID {gloss_id} not found")

    update_fields = []
    params = []
    if payload.gloss is not None:
        update_fields.append("gloss = %s")
        params.append(json.dumps(payload.gloss))
    if payload.confidence is not None:
        update_fields.append("confidence = %s")
        params.append(payload.confidence)
    if payload.inference_method is not None:
        update_fields.append("inference_method = %s")
        params.append(payload.inference_method)
    if payload.raw_openai_response is not None:
        update_fields.append("raw_openai_response = %s")
        params.append(json.dumps(payload.raw_openai_response))
    if payload.detected is not None:
        update_fields.append("detected = %s")
        params.append(payload.detected)
    if payload.landmarks is not None:
        update_fields.append("landmarks = %s")
        params.append(json.dumps(payload.landmarks))

    if not update_fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    params.append(str(gloss_id))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE video_gloss_requests SET {', '.join(update_fields)} WHERE id = %s",
                params,
            )

    # Fetch updated row
    updated_row = await fetch_video_gloss_by_id(pool, gloss_id)
    return _video_gloss_row_to_response(updated_row)


@app.delete("/video-gloss/{gloss_id}", tags=["video-gloss"])
async def delete_video_gloss(gloss_id: UUID, pool=Depends(get_pool)):
    if not pool:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    await delete_video_gloss_request(pool, gloss_id)
    return {"message": f"Video gloss {gloss_id} deleted successfully"}

# Video Gloss Prediction Route
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
    if not pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable; ensure DB env vars are set and proxy/DB is reachable",
        )
    print("[video-gloss] pool available, processing request")
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty video file."
        )
    landmarks_seq, method_used, detected_any = extract_landmarks_from_video(video_bytes, target_fps=target_fps)

    resolved_key = openai_key or os.getenv("OPENAI_API_KEY")
    gloss = ["HELLO", "THANK_YOU", "GOODBYE"]
    confidence = 0.5
    method = "fallback"
    raw_payload = None

    # Try OpenAI vision on a representative frame
    frame_b64 = extract_frame_base64(video_bytes)
    if resolved_key and OpenAI is not None and callable(OpenAI) and frame_b64:
        try:
            global _openai_client
            if _openai_client is None:
                _openai_client = OpenAI(api_key=resolved_key)
            client = _openai_client

            prompt = (
                "You are an ASL gloss predictor.\n"
                "Given an image of a hand sign, output JSON on one line:\n"
                '{ \"gloss\": \"<ASL_GLOSS>\", \"confidence\": <0-1> }\n'
                "Rules:\n"
                "- gloss uppercase with underscores\n"
                "- confidence between 0 and 1\n"
                "- no extra text\n"
            )

            vision_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=50,
                temperature=0.2,
            )
            vision_text = vision_resp.choices[0].message.content.strip()
            print(f"[video-gloss] raw OpenAI vision text: {vision_text!r}")
            try:
                payload = vision_text
                if payload.startswith("```"):
                    payload = payload.strip("`")
                    if payload.lower().startswith("json"):
                        payload = payload[4:]
                data = json.loads(payload)
                gloss_token = data.get("gloss")
                parsed_conf = float(data.get("confidence", 0.0))
                if not gloss_token:
                    raise ValueError("Missing gloss")
                gloss = [gloss_token]
                confidence = parsed_conf
                method = "openai-vision"
                raw_payload = vision_resp.model_dump()
            except Exception:
                gloss_match = re.search(r'"gloss"\s*:\s*"([^"]+)"', vision_text)
                conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', vision_text)
                gloss_token = gloss_match.group(1) if gloss_match else vision_text
                confidence = float(conf_match.group(1)) if conf_match else 0.5
                gloss = [gloss_token]
                method = "openai-vision"
                raw_payload = vision_resp.model_dump()
                print("[video-gloss] Parsed vision response via regex fallback")
        except Exception as e:
            print("[video-gloss] OpenAI vision inference failed:", repr(e))
            traceback.print_exc()

    # Fallback to landmarks aggregation if vision not used or failed
    if method == "fallback" and detected_any:
        gloss, confidence, method, raw_payload = infer_gloss_from_sequence(landmarks_seq, api_key=openai_key)

    req_id = str(uuid4())
    response = VideoGlossResponse(
        id=req_id,
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
            req_id=req_id,
            video_uri=file.filename,
            landmarks=landmarks_seq,
            gloss=gloss,
            confidence=confidence,
            method=response.inference_method,
            raw_openai=raw_payload,
            detected=detected_any,
        )
    except Exception:
        print("[video-gloss] Failed to save video gloss to DB; request not persisted.")

    return response

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

@app.get("/health")
async def health():
    """Simple health endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
