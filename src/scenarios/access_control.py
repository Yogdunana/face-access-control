"""Access control scenario - the original door entry system."""

from datetime import datetime
from typing import Any

from .base import BaseScenario


class AccessControlScenario(BaseScenario):
    """Door access control scenario with time-slot permission checking."""

    def __init__(self, user_manager=None, log_manager=None):
        self._user_manager = user_manager
        self._log_manager = log_manager

    @property
    def name(self) -> str:
        return "门禁控制"

    @property
    def description(self) -> str:
        return "基于人脸识别的门禁控制系统，支持多时段权限管理"

    def on_recognition_success(
        self, user_id: str, user_name: str, confidence: float
    ) -> str:
        if self._user_manager:
            allowed, reason = self._user_manager.check_time_permission(user_id)
            if not allowed:
                if self._log_manager:
                    self._log_manager.log_recognition_denied(user_name, reason)
                return f"⛔ 权限拒绝: {user_name} — {reason}"
        if self._log_manager:
            self._log_manager.log_recognition_success(user_name, confidence)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"✅ 门禁已开启: {user_name} (置信度: {confidence:.2%}) [{now}]"

    def on_recognition_failure(self) -> str:
        if self._log_manager:
            self._log_manager.log_recognition_failure()
        return "❌ 识别失败: 未匹配到已注册用户"

    def get_menu_actions(self) -> list[dict[str, Any]]:
        return [
            {"name": "时段管理", "description": "管理用户门禁时段权限", "action": "time_slot_management"},
            {"name": "通行记录", "description": "查看门禁通行记录", "action": "access_logs"},
        ]

    def get_dashboard_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"scenario": "access_control"}
        if self._user_manager:
            data["total_users"] = self._user_manager.user_count
        return data
