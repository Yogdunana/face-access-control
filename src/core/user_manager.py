"""
User management module for the Face Access Control Platform.
Handles user CRUD operations, face data management, and time-slot permissions.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


class UserManager:
    """Manages user data, face images, and access time slots."""

    def __init__(
        self,
        users_file: str = "data/users.json",
        face_data_dir: str = "data/face_data/",
        features_file: str = "data/models/face_features.npy",
    ):
        self._users_file = Path(users_file)
        self._face_data_dir = Path(face_data_dir)
        self._features_file = Path(features_file)
        self._face_features: dict[str, np.ndarray] = {}
        self._ensure_dirs()
        self._users: list[dict[str, Any]] = self._load_users()
        self._load_features()

    def _ensure_dirs(self) -> None:
        self._users_file.parent.mkdir(parents=True, exist_ok=True)
        self._face_data_dir.mkdir(parents=True, exist_ok=True)
        self._features_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_users(self) -> list[dict[str, Any]]:
        if self._users_file.exists():
            with open(self._users_file, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_users(self) -> None:
        with open(self._users_file, "w", encoding="utf-8") as f:
            json.dump(self._users, f, ensure_ascii=False, indent=2)

    def _load_features(self) -> None:
        if self._features_file.exists():
            try:
                data = np.load(self._features_file, allow_pickle=True).item()
                if isinstance(data, dict):
                    self._face_features = data
            except Exception:
                self._face_features = {}

    def _save_features(self) -> None:
        np.save(self._features_file, self._face_features)

    def list_users(self) -> list[dict[str, Any]]:
        """Return all users (without sensitive face image paths)."""
        return [
            {
                "id": u["id"],
                "name": u["name"],
                "face_count": len(u.get("face_images", [])),
                "time_slots": u.get("time_slots", []),
                "created_at": u.get("created_at", ""),
            }
            for u in self._users
        ]

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        for u in self._users:
            if u["id"] == user_id:
                return u
        return None

    def get_user_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a user by name."""
        for u in self._users:
            if u["name"] == name:
                return u
        return None

    def add_user(self, name: str, face_images: list[str] | None = None) -> tuple[bool, str]:
        """Add a new user."""
        if self.get_user_by_name(name):
            return False, f"用户名 '{name}' 已存在"
        user_id = str(uuid4())[:8]
        user = {
            "id": user_id,
            "name": name,
            "face_images": face_images or [],
            "time_slots": [],
            "created_at": datetime.now().isoformat(),
        }
        self._users.append(user)
        self._save_users()
        return True, f"用户 '{name}' 添加成功 (ID: {user_id})"

    def delete_user(self, user_id: str) -> tuple[bool, str]:
        """Delete a user and their face data."""
        user = self.get_user(user_id)
        if not user:
            return False, f"未找到 ID 为 {user_id} 的用户"
        for img_path in user.get("face_images", []):
            path = Path(img_path)
            if path.exists():
                path.unlink()
        self._face_features.pop(user_id, None)
        self._save_features()
        user_dir = self._face_data_dir / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir, ignore_errors=True)
        self._users = [u for u in self._users if u["id"] != user_id]
        self._save_users()
        return True, f"用户 '{user['name']}' 已删除"

    def delete_users_batch(self, user_ids: list[str]) -> tuple[int, list[str]]:
        """Delete multiple users by ID."""
        success = 0
        errors = []
        for uid in user_ids:
            ok, msg = self.delete_user(uid)
            if ok:
                success += 1
            else:
                errors.append(msg)
        return success, errors

    def update_user_name(self, user_id: str, new_name: str) -> tuple[bool, str]:
        """Update a user's display name."""
        user = self.get_user(user_id)
        if not user:
            return False, f"未找到 ID 为 {user_id} 的用户"
        existing = self.get_user_by_name(new_name)
        if existing and existing["id"] != user_id:
            return False, f"用户名 '{new_name}' 已存在"
        old_name = user["name"]
        user["name"] = new_name
        self._save_users()
        return True, f"用户名已从 '{old_name}' 更新为 '{new_name}'"

    def set_face_images(self, user_id: str, image_paths: list[str]) -> tuple[bool, str]:
        """Set face images for a user."""
        user = self.get_user(user_id)
        if not user:
            return False, f"未找到 ID 为 {user_id} 的用户"
        user["face_images"] = image_paths
        self._save_users()
        return True, f"用户 '{user['name']}' 的人脸图像已更新 ({len(image_paths)} 张)"

    def set_face_embedding(self, user_id: str, embedding: np.ndarray) -> None:
        """Store a face embedding (feature vector) for a user."""
        self._face_features[user_id] = embedding
        self._save_features()

    def get_face_embedding(self, user_id: str) -> np.ndarray | None:
        """Retrieve the face embedding for a user."""
        return self._face_features.get(user_id)

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return all stored face embeddings."""
        return self._face_features

    def has_face_data(self, user_id: str) -> bool:
        """Check if a user has registered face data."""
        user = self.get_user(user_id)
        if not user:
            return False
        return bool(user.get("face_images")) or user_id in self._face_features

    def check_face_duplicate(self, embedding: np.ndarray, threshold: float = 0.6) -> str | None:
        """Check if a face embedding matches any existing user."""
        for uid, stored_emb in self._face_features.items():
            if stored_emb is not None and len(stored_emb) == len(embedding):
                similarity = np.dot(embedding, stored_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_emb) + 1e-8
                )
                if similarity > threshold:
                    return uid
        return None

    def add_time_slot(
        self, user_id: str, start_time: str, end_time: str
    ) -> tuple[bool, str]:
        """Add an access time slot for a user."""
        user = self.get_user(user_id)
        if not user:
            return False, f"未找到 ID 为 {user_id} 的用户"
        slot_id = str(uuid4())[:8]
        slot = {"id": slot_id, "start_time": start_time, "end_time": end_time}
        user.setdefault("time_slots", []).append(slot)
        self._save_users()
        return True, f"已为用户 '{user['name']}' 添加时段 {start_time}-{end_time}"

    def remove_time_slot(self, user_id: str, slot_id: str) -> tuple[bool, str]:
        """Remove a time slot from a user."""
        user = self.get_user(user_id)
        if not user:
            return False, f"未找到 ID 为 {user_id} 的用户"
        slots = user.get("time_slots", [])
        original_len = len(slots)
        user["time_slots"] = [s for s in slots if s["id"] != slot_id]
        if len(user["time_slots"]) == original_len:
            return False, f"未找到时段 ID {slot_id}"
        self._save_users()
        return True, f"已删除时段 {slot_id}"

    def update_time_slot(
        self, user_id: str, slot_id: str, start_time: str, end_time: str
    ) -> tuple[bool, str]:
        """Update a time slot."""
        user = self.get_user(user_id)
        if not user:
            return False, f"未找到 ID 为 {user_id} 的用户"
        for slot in user.get("time_slots", []):
            if slot["id"] == slot_id:
                slot["start_time"] = start_time
                slot["end_time"] = end_time
                self._save_users()
                return True, f"时段已更新为 {start_time}-{end_time}"
        return False, f"未找到时段 ID {slot_id}"

    def check_time_permission(self, user_id: str) -> tuple[bool, str]:
        """Check if the current time falls within any of the user's authorized time slots."""
        user = self.get_user(user_id)
        if not user:
            return False, "用户不存在"
        slots = user.get("time_slots", [])
        if not slots:
            return True, "全天通行（未设置时段限制）"
        now = datetime.now().time()
        for slot in slots:
            start = datetime.strptime(slot["start_time"], "%H:%M").time()
            end = datetime.strptime(slot["end_time"], "%H:%M").time()
            if start <= now <= end:
                return True, f"在授权时段内 ({slot['start_time']}-{slot['end_time']})"
        slot_str = ", ".join(f"{s['start_time']}-{s['end_time']}" for s in slots)
        return False, f"当前时间不在授权时段内。授权时段: {slot_str}"

    @property
    def user_count(self) -> int:
        return len(self._users)
