from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    name: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    kind: str = "float"             # float | int | bool | choice
    choices: list[str] = field(default_factory=list)
    help: str = ""


class Tunable:
    """Mixin: parameters described by ``SPECS`` and stored in ``self.p``."""

    SPECS: list[ParamSpec] = []

    def __init__(self, **overrides):
        self.p: dict[str, Any] = {s.name: s.default for s in self.SPECS}
        for k, v in overrides.items():
            self.set(k, v)

    def set(self, name: str, value):
        spec = next((s for s in self.SPECS if s.name == name), None)
        if spec is None:
            raise KeyError(f"{type(self).__name__} has no parameter '{name}'")
        if spec.kind == "int":
            value = int(value)
        elif spec.kind == "float":
            value = float(value)
        elif spec.kind == "bool":
            value = bool(value)
        elif spec.kind == "choice" and value not in spec.choices:
            raise ValueError(f"{name}={value!r} not in {spec.choices}")
        self.p[name] = value
        self.on_param_change(name)

    def on_param_change(self, name: str):
        pass
