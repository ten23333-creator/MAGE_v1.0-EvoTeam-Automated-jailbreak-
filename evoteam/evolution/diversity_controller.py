"""
Tool diversity controller using embedding similarity.

Prevents tool population from converging to a single strategy
by applying fitness penalties to overly similar tools.
"""

import hashlib
from typing import Any, List, Dict
import numpy as np


class DiversityController:
    """Controls tool population diversity via embedding similarity penalties."""

    def __init__(
        self,
        embedding_model: Any = None,  # LocalModel for generating embeddings
        threshold: float = 0.85,  # Cosine similarity above which penalty is applied
        penalty_weight: float = 0.1,  # How much to penalize per similar pair
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.penalty_weight = penalty_weight

    def compute_similarity_matrix(self, tools: List[Any]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix for a population of tools."""
        n = len(tools)
        if n <= 1:
            return np.zeros((n, n))

        # Get embeddings for each tool
        embeddings = []
        for tool in tools:
            emb = self._get_tool_embedding(tool)
            embeddings.append(emb)

        # Compute cosine similarity matrix
        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def apply_diversity_penalties(self, tools: List[Any]):
        """Apply fitness penalties to tools that are too similar to others."""
        if len(tools) <= 1:
            return

        sim_matrix = self.compute_similarity_matrix(tools)

        for i, tool in enumerate(tools):
            # Count how many other tools are too similar
            similar_count = 0
            for j in range(len(tools)):
                if i != j and sim_matrix[i, j] > self.threshold:
                    similar_count += 1

            # Apply penalty proportional to number of similar neighbors
            tool.diversity_penalty = similar_count * self.penalty_weight

    def get_most_diverse_subset(self, tools: List[Any], k: int) -> List[Any]:
        """Select k most diverse tools from the population."""
        if len(tools) <= k:
            return list(tools)

        sim_matrix = self.compute_similarity_matrix(tools)
        n = len(tools)

        # Greedy selection: pick tool with lowest avg similarity to already selected
        selected_indices = [0]  # Start with the first tool
        remaining = set(range(1, n))

        while len(selected_indices) < k:
            best_idx = -1
            best_score = float("inf")

            for idx in remaining:
                # Average similarity to already selected tools
                avg_sim = np.mean([sim_matrix[idx, s] for s in selected_indices])
                if avg_sim < best_score:
                    best_score = avg_sim
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [tools[i] for i in selected_indices]

    def get_diversity_stats(self, tools: List[Any]) -> Dict:
        """Get population diversity statistics."""
        if len(tools) <= 1:
            return {
                "avg_pairwise_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "unique_approaches": len(set(getattr(t, 'approach', '') for t in tools)),
                "population_size": len(tools),
            }

        sim_matrix = self.compute_similarity_matrix(tools)

        # Get upper triangle (excluding diagonal)
        upper_tri = []
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                upper_tri.append(sim_matrix[i, j])

        approaches = set()
        for t in tools:
            approach = getattr(t, 'approach', '') or getattr(t, 'tool_category', '')
            approaches.add(approach)

        return {
            "avg_pairwise_similarity": float(np.mean(upper_tri)),
            "min_similarity": float(np.min(upper_tri)),
            "max_similarity": float(np.max(upper_tri)),
            "below_threshold_ratio": float(np.mean([s < self.threshold for s in upper_tri])),
            "unique_approaches": len(approaches),
            "approaches": list(approaches),
            "population_size": len(tools),
        }

    def _get_tool_embedding(self, tool: Any) -> List[float]:
        """Get embedding vector for a tool using its description and code summary."""
        # Build representative text
        text = (
            f"{getattr(tool, 'tool_name', '')} "
            f"{getattr(tool, 'tool_description', '')} "
            f"{getattr(tool, 'approach', '')} "
            f"{getattr(tool, 'tool_category', '')} "
            f"{getattr(tool, 'tool_code', '')[:500]}"  # First 500 chars of code
        )

        if self.embedding_model:
            try:
                return self.embedding_model.get_embedding(text)
            except Exception:
                pass

        # Fallback: simple bag-of-words hash (deterministic via SHA-256)
        words = text.lower().split()
        vec = np.zeros(128)
        for w in words:
            bucket = hashlib.sha256(w.encode()).digest()[0] % 128
            vec[bucket] += 1
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()