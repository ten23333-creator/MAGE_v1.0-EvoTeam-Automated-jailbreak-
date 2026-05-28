"""
Tagged dictionary cache for cross-query memory.

Stores successful tools and plans indexed by semantic query tags.
Provides O(1) lookup for warm-starting new attack sessions.
"""

import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime


class TaggedCache:
    """Cross-query memory via tagged dictionary cache."""

    def __init__(
        self,
        capacity: int = 10,
        persistence_path: Optional[str] = None,
    ):
        self.capacity = capacity
        self.persistence_path = persistence_path

        # Main cache structure: {tag: {tools, plans, best_approaches, ...}}
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Load from disk if available
        if persistence_path and os.path.exists(persistence_path):
            self._load()

    def _ensure_tag_entry(self, tag: str):
        """Create a cache entry for a tag if it doesn't exist."""
        if tag not in self._cache:
            self._cache[tag] = {
                "successful_tools": [],
                "successful_plans": [],
                "best_approaches": [],
                "avg_score": 0.0,
                "sample_count": 0,
                "last_updated": datetime.now().isoformat(),
            }

    def add_tool(self, tag: str, tool: Any):
        """Add a successful tool to the cache for the given tag."""
        self._ensure_tag_entry(tag)

        entry = self._cache[tag]
        tool_dict = tool.to_dict() if hasattr(tool, "to_dict") else tool

        # Avoid duplicates by tool_id
        existing_ids = {t.get("tool_id", "") for t in entry["successful_tools"]}
        tool_id = tool_dict.get("tool_id", "")
        if tool_id in existing_ids:
            return

        entry["successful_tools"].append(tool_dict)

        # Trim to capacity (keep highest-scoring)
        entry["successful_tools"] = sorted(
            entry["successful_tools"],
            key=lambda t: t.get("avg_judge_score", 0) or t.get("fitness", 0),
            reverse=True,
        )[:self.capacity]

        # Update stats
        entry["sample_count"] += 1
        self._update_tag_stats(tag)
        self._save()

    def add_plan(self, tag: str, plan: Any):
        """Add a successful attack plan to the cache."""
        self._ensure_tag_entry(tag)

        entry = self._cache[tag]
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan

        existing_ids = {p.get("plan_id", "") for p in entry["successful_plans"]}
        plan_id = plan_dict.get("plan_id", "")
        if plan_id in existing_ids:
            return

        entry["successful_plans"].append(plan_dict)
        entry["successful_plans"] = sorted(
            entry["successful_plans"],
            key=lambda p: p.get("success_count", 0) / max(1, p.get("total_attempts", 1)),
            reverse=True,
        )[:self.capacity]

        self._save()

    def add_approach(self, tag: str, approach: str):
        """Track a successful attack approach for this tag."""
        self._ensure_tag_entry(tag)
        entry = self._cache[tag]
        if approach not in entry["best_approaches"]:
            entry["best_approaches"].insert(0, approach)
            entry["best_approaches"] = entry["best_approaches"][:5]

    def get_tools(self, tag: str, n: int = 3) -> List[Dict]:
        """Get top N successful tools for a tag."""
        if tag not in self._cache:
            return []
        return self._cache[tag]["successful_tools"][:n]

    def get_plans(self, tag: str, n: int = 3) -> List[Dict]:
        """Get top N successful plans for a tag."""
        if tag not in self._cache:
            return []
        return self._cache[tag]["successful_plans"][:n]

    def get_best_approaches(self, tag: str) -> List[str]:
        """Get the most effective attack approaches for a tag."""
        if tag not in self._cache:
            return []
        return self._cache[tag]["best_approaches"]

    def get_stats(self, tag: str) -> Dict:
        """Get cache statistics for a tag."""
        if tag not in self._cache:
            return {"avg_score": 0.0, "sample_count": 0}
        return {
            "avg_score": self._cache[tag]["avg_score"],
            "sample_count": self._cache[tag]["sample_count"],
            "tools_count": len(self._cache[tag]["successful_tools"]),
            "plans_count": len(self._cache[tag]["successful_plans"]),
        }

    def has_tag(self, tag: str) -> bool:
        """Check if cache has data for a tag."""
        return tag in self._cache and self._cache[tag]["sample_count"] > 0

    def get_all_tags(self) -> List[str]:
        """Get all tags with cached data."""
        return sorted(self._cache.keys())

    def _update_tag_stats(self, tag: str):
        """Recalculate aggregate statistics for a tag."""
        entry = self._cache[tag]
        tools = entry["successful_tools"]
        if tools:
            scores = [t.get("avg_judge_score", 0) or t.get("fitness", 0) for t in tools]
            entry["avg_score"] = sum(scores) / len(scores)
        entry["last_updated"] = datetime.now().isoformat()

    def _save(self):
        """Persist cache to disk."""
        if not self.persistence_path:
            return
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False, default=str)

    def _load(self):
        """Load cache from disk."""
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self._cache = {}

    def to_dict(self) -> Dict:
        """Export entire cache as dictionary."""
        return self._cache

    def clear(self):
        """Clear all cached data."""
        self._cache = {}
        if self.persistence_path and os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)