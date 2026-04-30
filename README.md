# Face Access Control Platform

A pluggable, multi-scenario face recognition platform built with Python. Supports access control, attendance tracking, visitor management, and surveillance scenarios with swappable deep learning backends.

## Features

- **Pluggable Recognition Backends** — Switch between ArcFace (InsightFace), DeepFace models, and legacy LBPH via config
- **Multi-Scenario Support** — Access control, attendance, visitor management, and surveillance out of the box
- **RESTful API** — FastAPI-based HTTP API with WebSocket support for real-time video streaming
- **Security Hardened** — bcrypt password hashing, AES-256 encrypted face data, audit logging
- **CLI Interface** — Full-featured command-line admin console
- **Docker Ready** — CPU and GPU Docker images for easy deployment

## Quick Start

```bash
git clone https://github.com/Yogdunana/face-access-control.git
cd face-access-control

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m src.core.main
```

## Architecture

```
src/
├── core/               # Core modules
│   ├── recognizer.py   # Face detection & recognition (pluggable backends)
│   ├── user_manager.py # User CRUD & face data management
│   ├── auth.py         # Authentication & password security
│   ├── log_manager.py  # Audit logging
│   ├── console_ui.py   # CLI interface
│   └── main.py         # Entry point
├── scenarios/          # Pluggable scenario engines
│   ├── base.py         # Scenario base class
│   ├── access_control.py
│   ├── attendance.py
│   ├── visitor.py
│   └── surveillance.py
└── api/                # FastAPI web interface
    └── app.py
```

## Recognition Backends

| Backend | Accuracy (LFW) | Speed | GPU Required |
|---------|---------------|-------|-------------|
| ArcFace (InsightFace) | 99.83% | Fast | Optional |
| VGG-Face (DeepFace) | 98.95% | Medium | Recommended |
| FaceNet512 (DeepFace) | 99.65% | Medium | Recommended |
| SFace (DeepFace) | 99.60% | Fast | Optional |
| LBPH (OpenCV) | ~75% | Very Fast | No |

## License

MIT License - see [LICENSE](LICENSE) file for details.