"""
Session context for EvoTeam attack sessions.
Tracks state across iterations and queries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class SessionContext:
    """Unified session context tracking attack progress across iterations."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    query_tag: str = ""  # Semantic tag for current query

    # Model references (stored as dicts for serialization)
    target_model_config: Dict[str, Any] = field(default_factory=dict)
    judge_model_config: Dict[str, Any] = field(default_factory=dict)
    attack_model_config: Dict[str, Any] = field(default_factory=dict)
    embedding_model_config: Dict[str, Any] = field(default_factory=dict)

    # Session tracking
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    total_attacks: int = 0
    successful_attacks: int = 0
    current_iteration: int = 0
    current_phase: str = "fingerprinting"  # Current pipeline phase

    # Model fingerprint
    model_fingerprint: Dict[str, Any] = field(default_factory=dict)

    # Tool management
    created_tools: List[Any] = field(default_factory=list)
    tool_population: List[Any] = field(default_factory=list)  # GA population

    # Attack history
    attack_history: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-query cache reference
    tagged_cache: Optional[Any] = None

    # Flexible storage
    session_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.session_data.update({
            "session_id": self.session_id,
            "original_query": self.original_query,
            "query_tag": self.query_tag,
            "start_time": self.start_time,
            "total_attacks": self.total_attacks,
            "successful_attacks": self.successful_attacks,
        })

    def add_tool(self, tool: Any):
        """Add a tool to the context."""
        self.created_tools.append(tool)

    def add_attack_result(self, result: Dict[str, Any]):
        """Record an attack result."""
        self.attack_history.append(result)
        self.total_attacks += 1
        if result.get("attack_successful", False):
            self.successful_attacks += 1

    def set_phase(self, phase: str):
        """Set the current pipeline phase."""
        self.current_phase = phase

    def get_best_tools(self, n: int = 3) -> List[Any]:
        """Get top N tools by fitness."""
        sorted_tools = sorted(
            self.tool_population if self.tool_population else self.created_tools,
            key=lambda t: t.fitness if hasattr(t, 'fitness') else 0,
            reverse=True,
        )
        return sorted_tools[:n]

    @property
    def success_rate(self) -> float:
        if self.total_attacks == 0:
            return 0.0
        return self.successful_attacks / self.total_attacks