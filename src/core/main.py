"""
Main entry point for the Face Access Control Platform.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# noqa: E402 — imports must come after sys.path modification
from src.core.auth import AuthManager  # noqa: E402
from src.core.config import Config  # noqa: E402
from src.core.console_ui import ConsoleUI  # noqa: E402
from src.core.log_manager import LogManager  # noqa: E402
from src.core.recognizer import create_detector, create_recognizer  # noqa: E402
from src.core.user_manager import UserManager  # noqa: E402
from src.scenarios.access_control import AccessControlScenario  # noqa: E402
from src.scenarios.attendance import AttendanceScenario  # noqa: E402
from src.scenarios.base import registry  # noqa: E402
from src.scenarios.surveillance import SurveillanceScenario  # noqa: E402
from src.scenarios.visitor import VisitorScenario  # noqa: E402


def register_scenarios():
    """Register all available scenarios."""
    registry.register("access_control", AccessControlScenario)
    registry.register("attendance", AttendanceScenario)
    registry.register("visitor", VisitorScenario)
    registry.register("surveillance", SurveillanceScenario)


def admin_login(auth: AuthManager, ui: ConsoleUI) -> bool:
    """Handle admin login flow."""
    ui.show_banner()
    if not auth.password_changed:
        ui.display_warning("检测到默认密码未修改，建议登录后立即修改密码")
    for _ in range(3):
        username = ui.prompt("管理员用户名")
        if not username:
            username = "admin"
        password = ui.prompt_password()
        success, message = auth.authenticate(username, password)
        if success:
            ui.display_result(True, message)
            return True
        else:
            ui.display_result(False, message)
    ui.display_error("连续 3 次登录失败，系统退出")
    return False


def handle_face_recognition(config: Config, ui: ConsoleUI, scenario, log_manager: LogManager):
    """Handle the face recognition flow."""
    ui.display_info("正在启动摄像头...")
    try:
        import cv2
        backend = config.get("recognition", "backend", default="deepface")
        detector = create_detector(
            backend,
            detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
            model_path=config.get("recognition", "model_path", default="data/models/"),
        )
        device_index = config.get("camera", "device_index", default=0)
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            ui.display_error("无法打开摄像头，请检查设备连接")
            return
        detection_window = config.get("camera", "detection_window", default=5)
        ui.display_info(f"将在 {detection_window} 秒内进行人脸检测，请面对摄像头...")
        ui.countdown(detection_window, "检测中")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            ui.display_error("无法获取摄像头画面")
            return
        detections = detector.detect(frame)
        if not detections:
            result_msg = scenario.on_recognition_failure()
            ui.display_recognition_result(result_msg)
            return
        largest = max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])
        x, y, w, h = largest.bounding_box
        face_img = frame[y:y + h, x:x + w]
        recognizer = create_recognizer(
            backend,
            model_name=config.get("recognition", "deepface_model", default="ArcFace"),
            detector_backend=config.get("recognition", "deepface_detector", default="opencv"),
            model_path=config.get("recognition", "model_path", default="data/models/"),
        )
        embedding = recognizer.extract_embedding(face_img)
        if len(embedding) == 0:
            result_msg = scenario.on_recognition_failure()
            ui.display_recognition_result(result_msg)
            return
        user_manager = UserManager(
            users_file=config.get(
                "data", "users_file", default="data/users.json"
            ),
            face_data_dir=config.get(
                "face_collection", "face_data_dir", default="data/face_data/"
            ),
            features_file=config.get(
                "data", "features_file", default="data/models/face_features.npy"
            ),
        )
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
            result_msg = scenario.on_recognition_success(best_match, name, best_score)
        else:
            result_msg = scenario.on_recognition_failure()
        ui.display_recognition_result(result_msg)
    except ImportError:
        ui.display_error("缺少依赖库 (opencv-python)，请运行: pip install opencv-python")
    except Exception as e:
        ui.display_error(f"识别过程出错: {e}")


def handle_user_management(user_manager: UserManager, ui: ConsoleUI, log_manager: LogManager):
    """Handle user management submenu."""
    while True:
        ui.show_user_menu()
        choice = ui.prompt("请选择操作")
        if choice == "1":
            users = user_manager.list_users()
            ui.display_users(users)
        elif choice == "2":
            name = ui.prompt("请输入用户名")
            if name:
                ok, msg = user_manager.add_user(name)
                ui.display_result(ok, msg)
                if ok:
                    log_manager.log_user_added(name, "admin")
        elif choice == "3":
            user_id = ui.prompt("请输入要删除的用户 ID")
            if user_id:
                user = user_manager.get_user(user_id)
                if user:
                    confirm = ui.prompt(f"确认删除用户 '{user['name']}'? (y/n)")
                    if confirm.lower() == "y":
                        ok, msg = user_manager.delete_user(user_id)
                        ui.display_result(ok, msg)
                        if ok:
                            log_manager.log_user_deleted(user["name"], "admin")
                else:
                    ui.display_error(f"未找到 ID 为 {user_id} 的用户")
        elif choice == "4":
            ids_str = ui.prompt("请输入要删除的用户 ID（逗号分隔）")
            if ids_str:
                ids = [i.strip() for i in ids_str.split(",") if i.strip()]
                success, errors = user_manager.delete_users_batch(ids)
                ui.display_result(True, f"成功删除 {success} 个用户")
                for err in errors:
                    ui.display_warning(err)
        elif choice == "5":
            user_id = ui.prompt("请输入用户 ID")
            if user_id:
                new_name = ui.prompt("请输入新用户名")
                if new_name:
                    ok, msg = user_manager.update_user_name(user_id, new_name)
                    ui.display_result(ok, msg)
                    if ok:
                        log_manager.log_user_updated(new_name, "admin", "修改用户名")
        elif choice == "6":
            user_id = ui.prompt("请输入用户 ID")
            if user_id:
                user = user_manager.get_user(user_id)
                if not user:
                    ui.display_error(f"未找到 ID 为 {user_id} 的用户")
                    continue
                print(f"\n  用户: {user['name']}")
                slots = user.get("time_slots", [])
                if slots:
                    for s in slots:
                        print(f"    [{s['id']}] {s['start_time']} - {s['end_time']}")
                else:
                    print("    暂无时段设置（全天通行）")
                action = ui.prompt("操作: [a]添加 [d]删除 [u]更新")
                if action == "a":
                    start = ui.prompt("开始时间 (HH:MM)")
                    end = ui.prompt("结束时间 (HH:MM)")
                    ok, msg = user_manager.add_time_slot(user_id, start, end)
                    ui.display_result(ok, msg)
                elif action == "d":
                    slot_id = ui.prompt("时段 ID")
                    ok, msg = user_manager.remove_time_slot(user_id, slot_id)
                    ui.display_result(ok, msg)
                elif action == "u":
                    slot_id = ui.prompt("时段 ID")
                    start = ui.prompt("新开始时间 (HH:MM)")
                    end = ui.prompt("新结束时间 (HH:MM)")
                    ok, msg = user_manager.update_time_slot(user_id, slot_id, start, end)
                    ui.display_result(ok, msg)
        elif choice == "0":
            break
        ui.press_enter()


def handle_face_collection(
    config: Config,
    user_manager: UserManager,
    ui: ConsoleUI,
    log_manager: LogManager,
):
    """Handle face collection and registration."""
    ui.display_info("人脸录入功能需要摄像头支持")
    user_id = ui.prompt("请输入用户 ID")
    if not user_id:
        return
    user = user_manager.get_user(user_id)
    if not user:
        ui.display_error(f"未找到 ID 为 {user_id} 的用户")
        return
    try:
        import cv2
        capture_count = config.get("face_collection", "capture_count", default=100)
        save_count = config.get("face_collection", "save_count", default=10)
        face_data_dir = Path(
            config.get("face_collection", "face_data_dir", default="data/face_data/")
        )
        user_dir = face_data_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        device_index = config.get("camera", "device_index", default=0)
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            ui.display_error("无法打开摄像头")
            return
        ui.display_info(f"正在采集人脸，共需采集 {capture_count} 张...")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        collected = 0
        saved_paths = []
        while collected < capture_count:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                collected += 1
                if collected % (capture_count // save_count) == 0:
                    img_path = str(user_dir / f"face_{collected}.jpg")
                    face_img = frame[y:y + h, x:x + w]
                    cv2.imwrite(img_path, face_img)
                    saved_paths.append(img_path)
            cv2.imshow("Face Collection - Press Q to stop", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        if saved_paths:
            ok, msg = user_manager.set_face_images(user_id, saved_paths)
            ui.display_result(ok, msg)
            log_manager.log_face_registered(user["name"], len(saved_paths))
        else:
            ui.display_error("未采集到人脸图像")
    except ImportError:
        ui.display_error("缺少 opencv-python 依赖")


def handle_settings(config: Config, auth: AuthManager, ui: ConsoleUI):
    """Handle system settings."""
    print("\n── 系统设置 ──")
    print("  [1] 修改管理员密码")
    print(f"  [2] 当前识别后端: {config.get('recognition', 'backend')}")
    print(f"  [3] 当前场景: {config.get('scenario', 'type')}")
    print("  [0] 返回主菜单")
    choice = ui.prompt("请选择")
    if choice == "1":
        old_pwd = ui.prompt_password("请输入原密码")
        new_pwd = ui.prompt_password("请输入新密码")
        confirm_pwd = ui.prompt_password("请再次输入新密码")
        if new_pwd != confirm_pwd:
            ui.display_error("两次输入的新密码不一致")
            return
        ok, msg = auth.change_password(old_pwd, new_pwd)
        ui.display_result(ok, msg)
    elif choice == "2":
        ui.display_info(f"当前识别后端: {config.get('recognition', 'backend')}")
        ui.display_info("可通过修改 config.yaml 切换后端")


def main():
    """Main application entry point."""
    config_path = project_root / "config.yaml"
    config = Config(str(config_path) if config_path.exists() else None)
    auth = AuthManager(
        config_file=config.get("data", "admin_config_file", default="data/admin_config.json"),
        bcrypt_rounds=config.get("security", "bcrypt_rounds", default=12),
        max_attempts=config.get("security", "max_login_attempts", default=5),
        lockout_minutes=config.get("security", "lockout_minutes", default=30),
    )
    log_manager = LogManager(
        log_file=config.get("logging", "log_file", default="data/logs.json"),
        max_entries=config.get("logging", "max_entries", default=10000),
    )
    user_manager = UserManager(
        users_file=config.get("data", "users_file", default="data/users.json"),
        face_data_dir=config.get("face_collection", "face_data_dir", default="data/face_data/"),
        features_file=config.get("data", "features_file", default="data/models/face_features.npy"),
    )
    ui = ConsoleUI(config)
    register_scenarios()
    if not admin_login(auth, ui):
        return
    log_manager.log_login_success("admin")
    scenario_type = config.get("scenario", "type", default="access_control")
    scenario = registry.create(scenario_type, user_manager=user_manager, log_manager=log_manager)
    while True:
        ui.clear_screen()
        ui.show_banner()
        ui.display_info(f"当前场景: {scenario.name} — {scenario.description}")
        ui.show_main_menu()
        choice = ui.prompt("请选择操作")
        if choice == "1":
            handle_face_recognition(config, ui, scenario, log_manager)
        elif choice == "2":
            handle_user_management(user_manager, ui, log_manager)
        elif choice == "3":
            handle_face_collection(config, user_manager, ui, log_manager)
        elif choice == "4":
            scenarios = registry.list_scenarios()
            ui.show_scenario_menu(scenarios, scenario_type)
            sc_choice = ui.prompt("请选择场景")
            if sc_choice == "0":
                continue
            if sc_choice.isdigit():
                idx = int(sc_choice) - 1
                if 0 <= idx < len(scenarios):
                    scenario_type = scenarios[idx]["name"]
                    scenario = registry.create(
                        scenario_type, user_manager=user_manager, log_manager=log_manager
                    )
                    ui.display_result(True, f"已切换到场景: {scenario.name}")
        elif choice == "5":
            handle_settings(config, auth, ui)
        elif choice == "6":
            logs = log_manager.get_logs(limit=20)
            if logs:
                print("\n  最近日志:")
                for log in logs[-20:]:
                    icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
                    level_icon = icons.get(log.get("level", "info"), "ℹ️")
                    ts = log["timestamp"][:19]
                    evt = log["event_type"]
                    print(f"  {level_icon} [{ts}] [{evt}] {log['detail']}")
            else:
                ui.display_info("暂无日志记录")
        elif choice == "0":
            log_manager.log("system", "系统退出")
            ui.display_info("感谢使用，再见！")
            break
        ui.press_enter()


if __name__ == "__main__":
    main()
