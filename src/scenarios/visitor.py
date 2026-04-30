"""Visitor management scenario."""

from datetime import datetime
from typing import Any

from .base import BaseScenario


class VisitorScenario(BaseScenario):
    """Visitor management system with temporary access and appointment tracking."""

    def __init__(self, user_manager=None, log_manager=None):
        self._user_manager = user_manager
        self._log_manager = log_manager
        self._visitors: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "访客管理"

    @property
    def description(self) -> str:
        return "访客管理系统，支持预约登记、临时人脸录入和通行记录"

    def on_recognition_success(
        self, user_id: str, user_name: str, confidence: float
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._log_manager:
            self._log_manager.log_recognition_success(user_name, confidence)
        return f"✅ 访客通行: {user_name} (置信度: {confidence:.2%}) [{now}]"

    def on_recognition_failure(self) -> str:
        if self._log_manager:
            self._log_manager.log_recognition_failure("未知访客")
        return "❌ 识别失败: 未注册的访客，请先登记"

    def register_visitor(
        self,
        name: str,
        purpose: str = "",
        host: str = "",
        duration_minutes: int = 60,
    ) -> tuple[bool, str]:
        import uuid
        visitor_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        self._visitors[visitor_id] = {
            "id": visitor_id,
            "name": name,
            "purpose": purpose,
            "host": host,
            "registered_at": now.isoformat(),
            "expires_at": datetime.fromtimestamp(
                now.timestamp() + duration_minutes * 60
            ).isoformat(),
            "status": "waiting",
        }
        if self._log_manager:
            self._log_manager.log("visitor", f"访客 {name} 登记成功 (ID: {visitor_id})")
        return True, f"访客 '{name}' 登记成功 (ID: {visitor_id}, 有效期 {duration_minutes} 分钟)"

    def check_in_visitor(self, visitor_id: str) -> tuple[bool, str]:
        if visitor_id not in self._visitors:
            return False, "未找到该访客记录"
        visitor = self._visitors[visitor_id]
        if visitor["status"] != "waiting":
            return False, f"访客状态异常: {visitor['status']}"
        visitor["status"] = "checked_in"
        visitor["checked_in_at"] = datetime.now().isoformat()
        if self._log_manager:
            self._log_manager.log("visitor", f"访客 {visitor['name']} 已签到")
        return True, f"访客 '{visitor['name']}' 签到成功"

    def get_menu_actions(self) -> list[dict[str, Any]]:
        return [
            {"name": "访客登记", "description": "登记新访客信息", "action": "register_visitor"},
            {"name": "在访列表", "description": "查看当前在访的访客", "action": "active_visitors"},
            {"name": "访客记录", "description": "查看历史访客记录", "action": "visitor_history"},
        ]

    def get_dashboard_data(self) -> dict[str, Any]:
        active = sum(1 for v in self._visitors.values() if v["status"] == "checked_in")
        waiting = sum(1 for v in self._visitors.values() if v["status"] == "waiting")
        return {"scenario": "visitor", "active_visitors": active, "waiting_visitors": waiting, "total_today": len(self._visitors)}
