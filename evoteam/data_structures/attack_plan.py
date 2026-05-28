"""
Attack plan data structures.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class AttackPlan:
    """Pre-generated attack plan with persona, context, approach, and conversation plan."""

    plan_id: str
    tag: str  # Query category tag (e.g., "violence", "drugs")
    model_fingerprint_hash: str  # Target model fingerprint hash

    # Core plan components
    persona: str  # Attacker persona/role
    context: str  # Scenario context
    approach: str  # Attack strategy type
    conversation_plan: List[str] = field(default_factory=list)  # Multi-turn phase plan

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_query: Optional[str] = None  # Original query that inspired this plan
    success_count: int = 0  # Number of successful attacks using this plan
    total_attempts: int = 0  # Total attempts with this plan
    avg_score: float = 0.0  # Average judge score

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "tag": self.tag,
            "model_fingerprint_hash": self.model_fingerprint_hash,
            "persona": self.persona,
            "context": self.context,
            "approach": self.approach,
            "conversation_plan": self.conversation_plan,
            "created_at": self.created_at,
            "source_query": self.source_query,
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
            "avg_score": self.avg_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackPlan":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts