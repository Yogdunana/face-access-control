"""
Pluggable scenario engine base class.
Each scenario (access control, attendance, etc.) extends this base.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseScenario(ABC):
    """Abstract base class for all application scenarios."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this scenario does."""
        ...

    @abstractmethod
    def on_recognition_success(
        self,
        user_id: str,
        user_name: str,
        confidence: float,
    ) -> str:
        """Handle a successful face recognition event."""
        ...

    @abstractmethod
    def on_recognition_failure(self) -> str:
        """Handle a failed face recognition event (no match found)."""
        ...

    def get_menu_actions(self) -> list[dict[str, Any]]:
        return []

    def get_dashboard_data(self) -> dict[str, Any]:
        return {}


class ScenarioRegistry:
    """Registry for available scenarios."""

    def __init__(self):
        self._scenarios: dict[str, type[BaseScenario]] = {}

    def register(self, name: str, scenario_class: type[BaseScenario]) -> None:
        self._scenarios[name] = scenario_class

    def create(self, name: str, **kwargs: Any) -> BaseScenario:
        if name not in self._scenarios:
            available = ", ".join(self._scenarios.keys())
            raise ValueError(f"Unknown scenario: {name}. Available: {available}")
        return self._scenarios[name](**kwargs)

    def list_scenarios(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": cls.description}
            for name, cls in self._scenarios.items()
        ]


# Global registry
registry = ScenarioRegistry()
