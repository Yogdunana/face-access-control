# 人脸识别门禁平台

一个基于 Python 的可插拔、多场景人脸识别平台。支持门禁控制、考勤管理、访客管理和安防监控场景，可自由切换深度学习后端。

## 功能特性

- **可插拔识别后端** — 通过配置文件在 ArcFace (InsightFace)、DeepFace 模型和传统 LBPH 之间切换
- **多场景支持** — 开箱即用的门禁控制、考勤管理、访客管理、安防监控
- **RESTful API** — 基于 FastAPI 的 HTTP API，支持 WebSocket 实时视频流
- **安全加固** — bcrypt 密码哈希、AES-256 加密人脸数据、审计日志
- **CLI 界面** — 功能完整的命令行管理控制台
- **Docker 就绪** — 提供 CPU 和 GPU Docker 镜像，一键部署

## 快速开始

```bash
git clone https://github.com/Yogdunana/face-access-control.git
cd face-access-control

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m src.core.main
```

## 项目架构

```
src/
├── core/               # 核心模块
│   ├── recognizer.py   # 人脸检测与识别（可插拔后端）
│   ├── user_manager.py # 用户管理与人脸数据
│   ├── auth.py         # 认证与密码安全
│   ├── log_manager.py  # 审计日志
│   ├── console_ui.py   # 命令行界面
│   └── main.py         # 入口文件
├── scenarios/          # 可插拔场景引擎
│   ├── base.py         # 场景基类
│   ├── access_control.py  # 门禁控制
│   ├── attendance.py      # 考勤管理
│   ├── visitor.py         # 访客管理
│   └── surveillance.py    # 安防监控
└── api/                # FastAPI Web 接口
    └── app.py
```

## 识别后端对比

| 后端 | 准确率 (LFW) | 速度 | 需要 GPU |
|------|-------------|------|---------|
| ArcFace (InsightFace) | 99.83% | 快 | 可选 |
| VGG-Face (DeepFace) | 98.95% | 中 | 推荐 |
| FaceNet512 (DeepFace) | 99.65% | 中 | 推荐 |
| SFace (DeepFace) | 99.60% | 快 | 可选 |
| LBPH (OpenCV) | ~75% | 极快 | 否 |

## 配置说明

编辑 `config.yaml` 选择识别后端和场景：

```yaml
recognition:
  backend: "arcface"
  deepface_model: "ArcFace"
  confidence_threshold: 0.6

scenario:
  type: "access_control"

security:
  bcrypt_rounds: 12
  max_login_attempts: 5
  lockout_minutes: 30
```

## Docker 部署

```bash
docker compose up -d
```

## API 文档

启动服务器后访问：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 参与贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

## 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。