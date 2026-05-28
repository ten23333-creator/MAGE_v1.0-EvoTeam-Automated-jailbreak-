"""
Attack tool data structures for EvoTeam.

Represents AI-generated attack tools with execution tracking,
self-healing metadata, and performance history.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import uuid


@dataclass
class ToolPerformance:
    """Tracks execution and performance metrics for an attack tool."""

    execution_count: int = 0
    success_count: int = 0
    self_healing_count: int = 0  # Number of times self-healing was triggered
    self_healing_success_count: int = 0  # Successful self-healing repairs

    query_performance: List[Dict[str, Any]] = field(default_factory=list)
    judge_scores: List[float] = field(default_factory=list)

    average_execution_time: float = 0.0
    total_execution_time: float = 0.0

    @property
    def healing_success_rate(self) -> float:
        if self.self_healing_count == 0:
            return 0.0
        return self.self_healing_success_count / self.self_healing_count

    @property
    def execution_success_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def avg_judge_score(self) -> float:
        if not self.judge_scores:
            return 0.0
        return sum(self.judge_scores) / len(self.judge_scores)

    @property
    def best_judge_score(self) -> float:
        return max(self.judge_scores) if self.judge_scores else 0.0


@dataclass
class AttackTool:
    """Represents an AI-generated attack tool with code, metadata, and performance tracking."""

    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    tool_description: str = ""
    tool_category: str = ""
    tool_code: str = ""  # The Python code implementing the attack tool
    approach: str = ""  # Attack strategy category

    # Metadata
    created_for_query: str = ""  # Original query this tool was created for
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_count: int = 0  # Number of times this tool has been regenerated/fixed

    # Performance tracking
    performance: ToolPerformance = field(default_factory=ToolPerformance)

    # Evolution metadata
    parent_ids: List[str] = field(default_factory=list)  # Parent tool IDs (for GA crossover)
    generation: int = 0  # Generation number in genetic algorithm
    diversity_penalty: float = 0.0  # Penalty applied for being too similar to others

    def execute(self, query: str, context: dict, sandbox_fn: Callable) -> Any:
        """Execute the tool code in the sandbox environment."""
        return sandbox_fn(self.tool_code, query, context)

    def record_execution(
        self,
        success: bool,
        execution_time: float,
        judge_score: float = 0.0,
        query: str = "",
        self_healing_used: bool = False,
        self_healing_succeeded: bool = False,
    ):
        """Record an execution result."""
        self.performance.execution_count += 1
        if success:
            self.performance.success_count += 1

        self.performance.judge_scores.append(judge_score)
        self.performance.total_execution_time += execution_time
        self.performance.average_execution_time = (
            self.performance.total_execution_time / self.performance.execution_count
        )

        self.performance.query_performance.append({
            "query": query,
            "score": judge_score,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
        })

        if self_healing_used:
            self.performance.self_healing_count += 1
            if self_healing_succeeded:
                self.performance.self_healing_success_count += 1

    def get_performance_for_query(self, query: str) -> Optional[Dict]:
        """Get the most recent performance data for a specific query."""
        for perf in reversed(self.performance.query_performance):
            if perf.get("query") == query:
                return perf
        return None

    @property
    def fitness(self) -> float:
        """Calculate fitness score for genetic algorithm selection.

        Combines judge scores, execution reliability, and self-healing capability.
        """
        if self.performance.execution_count == 0:
            return 0.0

        w = {"judge_score": 0.6, "execution_success": 0.25, "self_healing_success": 0.15}

        judge_component = self.performance.avg_judge_score / 5.0  # Normalize to [0,1]
        exec_component = self.performance.execution_success_rate
        healing_component = (
            self.performance.healing_success_rate if self.performance.self_healing_count > 0 else 0.0
        )

        raw_fitness = (
            w["judge_score"] * judge_component
            + w["execution_success"] * exec_component
            + w["self_healing_success"] * healing_component
        )

        # Apply diversity penalty as multiplicative discount (max 30% reduction)
        diversity_discount = 1.0 - min(self.diversity_penalty * 0.3, 0.3)
        return max(0.0, raw_fitness * diversity_discount)

    def to_dict(self) -> dict:
        """Serialize tool to dictionary."""
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "tool_category": self.tool_category,
            "tool_code": self.tool_code,
            "approach": self.approach,
            "created_for_query": self.created_for_query,
            "created_at": self.created_at,
            "generation_count": self.generation_count,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "diversity_penalty": self.diversity_penalty,
            "performance": {
                "execution_count": self.performance.execution_count,
                "success_count": self.performance.success_count,
                "self_healing_count": self.performance.self_healing_count,
                "self_healing_success_count": self.performance.self_healing_success_count,
                "judge_scores": self.performance.judge_scores,
                "query_performance": self.performance.query_performance,
                "average_execution_time": self.performance.average_execution_time,
            },
            "fitness": self.fitness,
            "avg_judge_score": self.performance.avg_judge_score,
            "best_judge_score": self.performance.best_judge_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackTool":
        """Deserialize tool from dictionary."""
        perf_data = data.pop("performance", {})
        tool = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        tool.performance = ToolPerformance(
            execution_count=perf_data.get("execution_count", 0),
            success_count=perf_data.get("success_count", 0),
            self_healing_count=perf_data.get("self_healing_count", 0),
            self_healing_success_count=perf_data.get("self_healing_success_count", 0),
            judge_scores=perf_data.get("judge_scores", []),
            query_performance=perf_data.get("query_performance", []),
            average_execution_time=perf_data.get("average_execution_time", 0.0),
        )
        return tool