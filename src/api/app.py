"""
FastAPI application for the Face Access Control Platform.
Provides RESTful API and WebSocket support for face recognition operations.
"""

import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.auth import AuthManager
from src.core.config import Config
from src.core.log_manager import LogManager
from src.core.recognizer import create_detector, create_recognizer
from src.core.user_manager import UserManager
from src.scenarios.access_control import AccessControlScenario
from src.scenarios.attendance import AttendanceScenario
from src.scenarios.base import registry
from src.scenarios.surveillance import SurveillanceScenario
from src.scenarios.visitor import VisitorScenario

# ── App Initialization ─────────────────────────────────────────

app = FastAPI(
    title="Face Access Control API",
    description="RESTful API for the Face Access Control Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ────────────────────────────────────────────────────

config = Config()
auth = AuthManager(
    config_file=config.get("data", "admin_config_file", default="data/admin_config.json"),
    bcrypt_rounds=config.get("security", "bcrypt_rounds", default=12),
)
log_manager = LogManager(
    log_file=config.get("logging", "log_file", default="data/logs.json"),
)
user_manager = UserManager(
    users_file=config.get("data", "users_file", default="data/users.json"),
    face_data_dir=config.get("face_collection", "face_data_dir", default="data/face_data/"),
    features_file=config.get("data", "features_file", default="data/models/face_features.npy"),
)

# Register scenarios
registry.register("access_control", AccessControlScenario)
registry.register("attendance", AttendanceScenario)
registry.register("visitor", VisitorScenario)
registry.register("surveillance", SurveillanceScenario)

active_scenario = registry.create(
    config.get("scenario", "type", default="access_control"),
    user_manager=user_manager,
    log_manager=log_manager,
)


# ── Pydantic Models ────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str
    password_changed: bool = False


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class UserCreate(BaseModel):
    name: str


class UserResponse(BaseModel):
    id: str
    name: str
    face_count: int
    time_slots: list[dict[str, str]]
    created_at: str


class TimeSlotCreate(BaseModel):
    start_time: str
    end_time: str


class RecognizeResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    confidence: float = 0.0
    message: str


class ScenarioInfo(BaseModel):
    name: str
    description: str


class ScenarioSwitch(BaseModel):
    scenario_type: str


# ── Auth Endpoints ─────────────────────────────────────────────


@app.post("/api/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    success, message = auth.authenticate(req.username, req.password)
    return LoginResponse(
        success=success,
        message=message,
        password_changed=auth.password_changed,
    )


@app.post("/api/auth/change-password")
async def change_password(req: ChangePasswordRequest):
    ok, msg = auth.change_password(req.old_password, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


# ── User Endpoints ─────────────────────────────────────────────


@app.get("/api/users", response_model=list[UserResponse])
async def list_users():
    return user_manager.list_users()


@app.post("/api/users", status_code=201)
async def create_user(req: UserCreate):
    ok, msg = user_manager.add_user(req.name)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    log_manager.log_user_added(req.name, "api")
    return {"success": True, "message": msg}


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    ok, msg = user_manager.delete_user(user_id)
    if ok:
        log_manager.log_user_deleted(user["name"], "api")
    return {"success": ok, "message": msg}


@app.put("/api/users/{user_id}/name")
async def update_user_name(user_id: str, name: str):
    ok, msg = user_manager.update_user_name(user_id, name)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


# ── Time Slot Endpoints ────────────────────────────────────────


@app.post("/api/users/{user_id}/time-slots")
async def add_time_slot(user_id: str, slot: TimeSlotCreate):
    ok, msg = user_manager.add_time_slot(user_id, slot.start_time, slot.end_time)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.delete("/api/users/{user_id}/time-slots/{slot_id}")
async def remove_time_slot(user_id: str, slot_id: str):
    ok, msg = user_manager.remove_time_slot(user_id, slot_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


# ── Face Registration Endpoint ─────────────────────────────────



@app.post("/api/users/{user_id}/register-face")
async def register_face(user_id: str, file: UploadFile = File(...)):
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片文件")

    # Detect face
    backend = config.get("recognition", "backend", default="deepface")
    detector = create_detector(
        backend,
        detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
        model_path=config.get("recognition", "model_path", default="data/models/"),
    )
    detections = detector.detect(image)
    if not detections:
        raise HTTPException(status_code=400, detail="未在图片中检测到人脸")

    # Use largest face
    largest = max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])
    x, y, w, h = largest.bounding_box
    face_img = image[y:y + h, x:x + w]

    # Extract embedding
    recognizer = create_recognizer(
        backend,
        model_name=config.get("recognition", "deepface_model", default="ArcFace"),
        detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
        model_path=config.get("recognition", "model_path", default="data/models/"),
    )
    embedding = recognizer.extract_embedding(face_img)
    if len(embedding) == 0:
        raise HTTPException(status_code=500, detail="人脸特征提取失败")

    # Check for duplicate
    threshold = config.get("recognition", "confidence_threshold", default=0.6)
    duplicate_id = user_manager.check_face_duplicate(embedding, threshold=threshold)
    if duplicate_id and duplicate_id != user_id:
        dup_user = user_manager.get_user(duplicate_id)
        dup_name = dup_user["name"] if dup_user else duplicate_id
        raise HTTPException(status_code=400, detail=f"该人脸已注册为用户 '{dup_name}'")

    # Save face image and embedding
    face_data_dir = Path(config.get("face_collection", "face_data_dir", default="data/face_data/"))
    user_dir = face_data_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(user_dir / f"face_{len(user.get('face_images', []))}.jpg")
    cv2.imwrite(img_path, face_img)

    user_manager.set_face_images(user_id, user.get("face_images", []) + [img_path])
    user_manager.set_face_embedding(user_id, embedding)
    log_manager.log_face_registered(user["name"], 1)

    return {"success": True, "message": f"人脸注册成功: {user['name']}"}


# ── Recognition Endpoint ───────────────────────────────────────


@app.post("/api/recognize", response_model=RecognizeResponse)
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片文件")

    backend = config.get("recognition", "backend", default="deepface")
    detector = create_detector(
        backend,
        detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
        model_path=config.get("recognition", "model_path", default="data/models/"),
    )
    recognizer = create_recognizer(
        backend,
        model_name=config.get("recognition", "deepface_model", default="ArcFace"),
        detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
        model_path=config.get("recognition", "model_path", default="data/models/"),
    )

    detections = detector.detect(image)
    if not detections:
        msg = active_scenario.on_recognition_failure()
        return RecognizeResponse(success=False, message=msg)

    largest = max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])
    x, y, w, h = largest.bounding_box
    face_img = image[y:y + h, x:x + w]

    embedding = recognizer.extract_embedding(face_img)
    if len(embedding) == 0:
        msg = active_scenario.on_recognition_failure()
        return RecognizeResponse(success=False, message=msg)

    threshold = config.get("recognition", "confidence_threshold", default=0.6)
    best_match = None
    best_score = 0.0

    for uid, stored_emb in user_manager.get_all_embeddings().items():
        if stored_emb is not None and len(stored_emb) == len(embedding):
            score = recognizer.compare(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_match = uid

    if best_match and best_score >= threshold:
        user = user_manager.get_user(best_match)
        name = user["name"] if user else best_match
        msg = active_scenario.on_recognition_success(best_match, name, best_score)
        return RecognizeResponse(
            success=True,
            user_id=best_match,
            user_name=name,
            confidence=best_score,
            message=msg,
        )
    else:
        msg = active_scenario.on_recognition_failure()
        return RecognizeResponse(success=False, confidence=best_score, message=msg)


# ── Scenario Endpoints ─────────────────────────────────────────



@app.get("/api/scenarios", response_model=list[ScenarioInfo])
async def list_scenarios():
    return [ScenarioInfo(**s) for s in registry.list_scenarios()]


@app.post("/api/scenarios/switch")
async def switch_scenario(req: ScenarioSwitch):
    global active_scenario
    try:
        active_scenario = registry.create(
            req.scenario_type,
            user_manager=user_manager,
            log_manager=log_manager,
        )
        return {"success": True, "message": f"已切换到场景: {active_scenario.name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/scenarios/dashboard")
async def get_dashboard():
    return active_scenario.get_dashboard_data()


# ── Log Endpoints ──────────────────────────────────────────────


@app.get("/api/logs")
async def get_logs(event_type: Optional[str] = None, limit: int = 100):
    return log_manager.get_logs(event_type=event_type, limit=limit)



# ── WebSocket for Real-time Recognition ────────────────────────


@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    await websocket.accept()
    try:
        backend = config.get("recognition", "backend", default="deepface")
        detector = create_detector(
            backend,
            detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
            model_path=config.get("recognition", "model_path", default="data/models/"),
        )
        recognizer = create_recognizer(
            backend,
            model_name=config.get("recognition", "deepface_model", default="ArcFace"),
            detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
            model_path=config.get("recognition", "model_path", default="data/models/"),
        )
        threshold = config.get("recognition", "confidence_threshold", default=0.6)

        while True:
            data = await websocket.receive_text()
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    await websocket.send_json({"success": False, "message": "无法解析图片"})
                    continue

                detections = detector.detect(image)
                if not detections:
                    msg = active_scenario.on_recognition_failure()
                    await websocket.send_json({"success": False, "message": msg})
                    continue

                largest = max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])
                x, y, w, h = largest.bounding_box
                face_img = image[y:y + h, x:x + w]

                embedding = recognizer.extract_embedding(face_img)
                if len(embedding) == 0:
                    msg = active_scenario.on_recognition_failure()
                    await websocket.send_json({"success": False, "message": msg})
                    continue

                best_match = None
                best_score = 0.0
                for uid, stored_emb in user_manager.get_all_embeddings().items():
                    if stored_emb is not None and len(stored_emb) == len(embedding):
                        score = recognizer.compare(embedding, stored_emb)
                        if score > best_score:
                            best_score = score
                            best_match = uid

                if best_match and best_score >= threshold:
                    user = user_manager.get_user(best_match)
                    name = user["name"] if user else best_match
                    msg = active_scenario.on_recognition_success(best_match, name, best_score)
                    await websocket.send_json({
                        "success": True,
                        "user_id": best_match,
                        "user_name": name,
                        "confidence": best_score,
                        "message": msg,
                    })
                else:
                    msg = active_scenario.on_recognition_failure()
                    await websocket.send_json({"success": False, "message": msg})

            except Exception as e:
                await websocket.send_json({"success": False, "message": str(e)})

    except WebSocketDisconnect:
        pass


# ── Health Check ───────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "backend": config.get("recognition", "backend"),
        "scenario": config.get("scenario", "type"),
        "users": user_manager.user_count,
    }
