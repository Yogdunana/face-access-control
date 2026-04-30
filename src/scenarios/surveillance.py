"""Surveillance and security monitoring scenario."""

from datetime import datetime
from typing import Any

from .base import BaseScenario


class SurveillanceScenario(BaseScenario):
    """Intelligent surveillance with stranger detection and alerting."""

    def __init__(self, user_manager=None, log_manager=None):
        self._user_manager = user_manager
        self._log_manager = log_manager
        self._alerts: list[dict[str, Any]] = []
        self._stranger_cooldown: dict[str, float] = {}
        self._alert_cooldown_seconds = 30

    @property
    def name(self) -> str:
        return "安防监控"

    @property
    def description(self) -> str:
        return "智能安防监控系统，支持陌生人检测告警和人员追踪"

    def on_recognition_success(
        self, user_id: str, user_name: str, confidence: float
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._log_manager:
            self._log_manager.log_recognition_success(user_name, confidence)
        return f"👤 已识别: {user_name} (置信度: {confidence:.2%}) [{now}]"

    def on_recognition_failure(self) -> str:
        import time
        now = time.time()
        camera_key = "default"
        last_alert = self._stranger_cooldown.get(camera_key, 0)
        if now - last_alert < self._alert_cooldown_seconds:
            return "⚠️ 未知人员 (告警冷却中)"
        self._stranger_cooldown[camera_key] = now
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": "stranger_detected",
            "camera": camera_key,
            "level": "warning",
        }
        self._alerts.append(alert)
        if self._log_manager:
            self._log_manager.log("surveillance", "检测到未知人员", level="warning")
        return "🚨 告警: 检测到未知人员！"

    def get_menu_actions(self) -> list[dict[str, Any]]:
        return [
            {"name": "告警记录", "description": "查看安防告警历史", "action": "alert_history"},
            {"name": "实时监控", "description": "开启实时监控模式", "action": "live_monitor"},
            {"name": "人员统计", "description": "查看区域人员统计", "action": "people_count"},
        ]

    def get_dashboard_data(self) -> dict[str, Any]:
        recent_alerts = [
            a for a in self._alerts
            if (datetime.now() - datetime.fromisoformat(a["timestamp"])).total_seconds() < 3600
        ]
        return {
            "scenario": "surveillance",
            "alerts_last_hour": len(recent_alerts),
            "total_alerts": len(self._alerts),
            "active_cameras": 1,
        }

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._alerts[-limit:]
