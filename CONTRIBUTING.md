# Contributing to Face Access Control Platform

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Fork and clone** the repository
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run linting: `ruff check src/ tests/`
5. Format code: `ruff format src/ tests/`
6. Commit with conventional commit messages
7. Push and create a Pull Request

## Code Style

- We use **ruff** for linting and formatting
- Type hints are encouraged (use **mypy** for checking)
- Maximum line length: 100 characters
- Follow PEP 8 conventions

## Adding a New Recognition Backend

1. Create a new class in `src/core/recognizer.py` that extends `FaceDetector` and/or `FaceRecognizer`
2. Implement all abstract methods
3. Register it in the `create_detector()` and `create_recognizer()` factory functions
4. Add tests in `tests/test_recognizer.py`
5. Update `config.yaml` documentation

## Adding a New Scenario

1. Create a new file in `src/scenarios/` extending `BaseScenario`
2. Implement `on_recognition_success()` and `on_recognition_failure()`
3. Optionally implement `get_menu_actions()` and `get_dashboard_data()`
4. Register in `src/core/main.py` and `src/api/app.py`
5. Add tests

## Pull Request Guidelines

- Keep PRs small and focused
- Include tests for new functionality
- Update documentation (README, docstrings)
- Ensure all CI checks pass
