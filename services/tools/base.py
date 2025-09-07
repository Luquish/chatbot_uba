from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Decision:
    score: float
    params: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    response: str
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(Protocol):
    name: str
    priority: int
    config: Dict[str, Any]

    def configure(self, config: Dict[str, Any]) -> None: ...
    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision: ...
    def accepts(self, score: float) -> bool: ...
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult: ...


