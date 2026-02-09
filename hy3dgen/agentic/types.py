from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

PlanStepType = Literal[
    "import",
    "normalize",
    "partgraph",
    "preset",
    "clean",
    "op",
    "validate",
    "export",
]


@dataclass
class PlanStep:
    id: str
    type: PlanStepType
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    op_class: Optional[str] = None
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "params": self.params,
            "enabled": self.enabled,
            "depends_on": self.depends_on,
        }
        if self.op_class is not None:
            payload["op_class"] = self.op_class
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanStep":
        return cls(
            id=str(payload["id"]),
            type=payload["type"],
            name=str(payload.get("name", payload["type"])),
            params=dict(payload.get("params", {})),
            op_class=payload.get("op_class"),
            enabled=bool(payload.get("enabled", True)),
            depends_on=list(payload.get("depends_on", [])),
        )


@dataclass
class PlanIR:
    version: str
    prompt: str
    preset: str
    seed: int
    steps: List[PlanStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "prompt": self.prompt,
            "preset": self.preset,
            "seed": self.seed,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanIR":
        return cls(
            version=str(payload.get("version", "1.0")),
            prompt=str(payload.get("prompt", "")),
            preset=str(payload.get("preset", "generic")),
            seed=int(payload.get("seed", 0)),
            steps=[PlanStep.from_dict(s) for s in payload.get("steps", [])],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ExecutionResult:
    success: bool
    plan: Dict[str, Any]
    artifacts: Dict[str, str]
    validation: Dict[str, Any]
    runtime: Dict[str, Any]
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "plan": self.plan,
            "artifacts": self.artifacts,
            "validation": self.validation,
            "runtime": self.runtime,
            "trace": self.trace,
        }
