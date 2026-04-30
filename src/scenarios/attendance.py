"""Attendance tracking scenario."""

from datetime import datetime, time
from typing import Any

from .base import BaseScenario


class AttendanceScenario(BaseScenario):
    """Smart attendance system with check-in/check-out and statistics."""

    def __init__(self, user_manager=None, log_manager=None):
        self._user_manager = user_manager
        self._log_manager = log_manager
        self._attendance: dict[str, dict[str, dict[str, str]]] = {}
        self._work_start = time(9, 0)
        self._work_end = time(18, 0)

    @property
    def name(self) -> str:
        return "考勤管理"

    @property
    def description(self) -> str:
        return "智能考勤系统，支持自动签到/签退和迟到早退统计"

    def on_recognition_success(
        self, user_id: str, user_name: str, confidence: float
    ) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().time()
        if user_id not in self._attendance:
            self._attendance[user_id] = {}
        today_record = self._attendance[user_id].get(today, {})
        if "check_in" not in today_record:
            is_late = now_time > self._work_start
            today_record["check_in"] = datetime.now().strftime("%H:%M:%S")
            today_record["late"] = str(is_late)
            self._attendance[user_id][today] = today_record
            status = "⚠️ 迟到签到" if is_late else "✅ 正常签到"
            if self._log_manager:
                self._log_manager.log("attendance", f"{status}: {user_name} ({today_record['check_in']})")
            return f"{status}: {user_name} — 签到时间 {today_record['check_in']}"
        elif "check_out" not in today_record:
            is_early = now_time < self._work_end
            today_record["check_out"] = datetime.now().strftime("%H:%M:%S")
            today_record["early_leave"] = str(is_early)
            self._attendance[user_id][today] = today_record
            status = "⚠️ 早退签退" if is_early else "✅ 正常签退"
            if self._log_manager:
                self._log_manager.log("attendance", f"{status}: {user_name} ({today_record['check_out']})")
            return f"{status}: {user_name} — 签退时间 {today_record['check_out']}"
        else:
            return f"ℹ️ {user_name} 今日已完成签到和签退"

    def on_recognition_failure(self) -> str:
        return "❌ 识别失败: 未匹配到已注册用户，无法记录考勤"

    def get_menu_actions(self) -> list[dict[str, Any]]:
        return [
            {"name": "今日考勤", "description": "查看今日所有用户考勤状态", "action": "today_attendance"},
            {"name": "考勤统计", "description": "查看月度考勤统计报表", "action": "attendance_stats"},
            {"name": "导出报表", "description": "导出考勤数据为 CSV", "action": "export_attendance"},
        ]

    def get_dashboard_data(self) -> dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        checked_in = sum(
            1 for records in self._attendance.values()
            if today in records and "check_in" in records[today]
        )
        late_count = sum(
            1 for records in self._attendance.values()
            if today in records and records[today].get("late") == "True"
        )
        return {"scenario": "attendance", "today_checked_in": checked_in, "today_late": late_count}

    def get_today_attendance(self) -> list[dict[str, Any]]:
        today = datetime.now().strftime("%Y-%m-%d")
        records = []
        for user_id, dates in self._attendance.items():
            if today in dates:
                user = self._user_manager.get_user(user_id) if self._user_manager else None
                name = user["name"] if user else user_id
                record = dates[today]
                records.append({
                    "user_id": user_id,
                    "user_name": name,
                    "check_in": record.get("check_in", "-"),
                    "check_out": record.get("check_out", "-"),
                    "late": record.get("late") == "True",
                    "early_leave": record.get("early_leave") == "True",
                })
        return records
