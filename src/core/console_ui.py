"""
Command-line interface for the Face Access Control Platform.
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class ConsoleUI:
    """Interactive command-line user interface."""

    def __init__(self, config: "Config"):
        self._config = config

    def clear_screen(self) -> None:
        print("\033[2J\033[H", end="")

    def show_banner(self) -> None:
        banner = """
╔══════════════════════════════════════════════════════╗
║        人脸识别门禁平台 Face Access Control          ║
║              v0.1.0 — Open Source Edition            ║
╚══════════════════════════════════════════════════════╝
        """
        print(banner)

    def show_main_menu(self) -> None:
        backend = self._config.get("recognition", "backend", default="deepface")
        scenario = self._config.get("scenario", "type", default="access_control")
        print(f"  识别后端: {backend}  |  当前场景: {scenario}")
        print("─" * 50)
        print("  [1] 刷脸识别")
        print("  [2] 用户管理")
        print("  [3] 人脸录入")
        print("  [4] 场景管理")
        print("  [5] 系统设置")
        print("  [6] 查看日志")
        print("  [0] 退出系统")
        print("─" * 50)

    def show_user_menu(self) -> None:
        print("\n── 用户管理 ──")
        print("  [1] 查看所有用户")
        print("  [2] 添加用户")
        print("  [3] 删除用户")
        print("  [4] 批量删除用户")
        print("  [5] 修改用户名")
        print("  [6] 时段管理")
        print("  [0] 返回主菜单")

    def show_scenario_menu(self, scenarios: list[dict[str, str]], current: str) -> None:
        print("\n── 场景管理 ──")
        for i, s in enumerate(scenarios, 1):
            marker = " ◀ 当前" if s["name"] == current else ""
            print(f"  [{i}] {s['name']} — {s['description']}{marker}")
        print("  [0] 返回主菜单")

    def prompt(self, message: str) -> str:
        return input(f"  {message}: ").strip()

    def prompt_password(self, message: str = "请输入密码") -> str:
        import getpass
        return getpass.getpass(f"  {message}: ")

    def prompt_choice(self, message: str, options: list[str]) -> str:
        print(f"  {message}:")
        for i, opt in enumerate(options, 1):
            print(f"    [{i}] {opt}")
        while True:
            choice = input("  请选择: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            print("  无效选择，请重试")

    def display_users(self, users: list[dict]) -> None:
        if not users:
            print("\n  暂无注册用户")
            return
        print(f"\n  {'ID':<8} {'姓名':<12} {'人脸数':<8} {'时段数':<8} {'创建时间':<20}")
        print("  " + "─" * 60)
        for u in users:
            slots = u.get("time_slots", [])
            print(
                f"  {u['id']:<8} {u['name']:<12} {u['face_count']:<8} "
                f"{len(slots):<8} {u.get('created_at', '-')[:19]:<20}"
            )
        print(f"\n  共 {len(users)} 名用户")

    def display_result(self, success: bool, message: str) -> None:
        icon = "✅" if success else "❌"
        print(f"\n  {icon} {message}")

    def display_info(self, message: str) -> None:
        print(f"\n  ℹ️  {message}")

    def display_warning(self, message: str) -> None:
        print(f"\n  ⚠️  {message}")

    def display_error(self, message: str) -> None:
        print(f"\n  ❌ 错误: {message}")

    def display_recognition_result(self, message: str) -> None:
        print(f"\n  {'═' * 46}")
        print(f"  {message}")
        print(f"  {'═' * 46}")

    def countdown(self, seconds: int, message: str = "检测中") -> None:
        for i in range(seconds, 0, -1):
            print(f"\r  {message}... {i}s", end="", flush=True)
            time.sleep(1)
        print("\r  " + " " * 30 + "\r", end="")

    def press_enter(self) -> None:
        input("\n  按 Enter 键继续...")
