"""Tests for the UserManager module."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.user_manager import UserManager


class TestUserManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.users_file = Path(self.tmpdir) / "users.json"
        self.face_dir = Path(self.tmpdir) / "faces"
        self.features_file = Path(self.tmpdir) / "features.npy"
        self.mgr = UserManager(
            users_file=str(self.users_file),
            face_data_dir=str(self.face_dir),
            features_file=str(self.features_file),
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_user(self):
        ok, msg = self.mgr.add_user("张三")
        assert ok is True
        assert self.mgr.user_count == 1

    def test_duplicate_name(self):
        self.mgr.add_user("张三")
        ok, msg = self.mgr.add_user("张三")
        assert ok is False
        assert "已存在" in msg

    def test_delete_user(self):
        ok, _ = self.mgr.add_user("张三")
        user = self.mgr.get_user_by_name("张三")
        ok, msg = self.mgr.delete_user(user["id"])
        assert ok is True
        assert self.mgr.user_count == 0

    def test_delete_nonexistent_user(self):
        ok, msg = self.mgr.delete_user("nonexistent")
        assert ok is False

    def test_update_name(self):
        self.mgr.add_user("张三")
        user = self.mgr.get_user_by_name("张三")
        ok, msg = self.mgr.update_user_name(user["id"], "李四")
        assert ok is True
        assert self.mgr.get_user_by_name("李四") is not None

    def test_batch_delete(self):
        self.mgr.add_user("张三")
        self.mgr.add_user("李四")
        self.mgr.add_user("王五")
        ids = [u["id"] for u in self.mgr.list_users()[:2]]
        success, errors = self.mgr.delete_users_batch(ids)
        assert success == 2
        assert self.mgr.user_count == 1

    def test_time_slots(self):
        self.mgr.add_user("张三")
        user = self.mgr.get_user_by_name("张三")
        ok, _ = self.mgr.add_time_slot(user["id"], "09:00", "18:00")
        assert ok is True
        ok, _ = self.mgr.add_time_slot(user["id"], "20:00", "22:00")
        assert ok is True
        user = self.mgr.get_user(user["id"])
        assert len(user["time_slots"]) == 2

    def test_face_embedding(self):
        self.mgr.add_user("张三")
        user = self.mgr.get_user_by_name("张三")
        embedding = np.random.randn(512).astype(np.float32)
        self.mgr.set_face_embedding(user["id"], embedding)
        retrieved = self.mgr.get_face_embedding(user["id"])
        assert retrieved is not None
        assert np.allclose(retrieved, embedding)

    def test_face_duplicate_check(self):
        self.mgr.add_user("张三")
        user = self.mgr.get_user_by_name("张三")
        embedding = np.random.randn(512).astype(np.float32)
        self.mgr.set_face_embedding(user["id"], embedding)
        duplicate_id = self.mgr.check_face_duplicate(embedding, threshold=0.9)
        assert duplicate_id == user["id"]
        different = np.random.randn(512).astype(np.float32)
        duplicate_id = self.mgr.check_face_duplicate(different, threshold=0.9)
        assert duplicate_id is None

    def test_list_users_no_images(self):
        self.mgr.add_user("张三")
        users = self.mgr.list_users()
        assert len(users) == 1
        assert "face_images" not in users[0]
