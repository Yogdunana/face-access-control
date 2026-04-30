"""Tests for the LogManager module."""
import tempfile

from src.core.log_manager import LogManager


class TestLogManager:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.tmpfile.close()
        self.log = LogManager(log_file=self.tmpfile.name, max_entries=100)

    def teardown_method(self):
        import os
        os.unlink(self.tmpfile.name)

    def test_log_entry(self):
        self.log.log("test", "test message")
        logs = self.log.get_logs()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "test"
        assert logs[0]["detail"] == "test message"

    def test_log_with_filters(self):
        self.log.log("login", "login success")
        self.log.log("recognition", "face detected")
        self.log.log("login", "login failure")
        login_logs = self.log.get_logs(event_type="login")
        assert len(login_logs) == 2

    def test_log_rotation(self):
        small_log = LogManager(log_file=self.tmpfile.name, max_entries=5)
        for i in range(10):
            small_log.log("test", f"message {i}")
        logs = small_log.get_logs(limit=100)
        assert len(logs) == 5

    def test_convenience_methods(self):
        self.log.log_login_success("admin")
        self.log.log_recognition_success("张三", 0.95)
        self.log.log_recognition_failure()
        logs = self.log.get_logs()
        assert len(logs) == 3

    def test_clear_logs(self):
        self.log.log("test", "message")
        count = self.log.clear_logs()
        assert count == 1
        assert len(self.log.get_logs()) == 0
