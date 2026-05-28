"""
Retrieval-Augmented Attack Memory with vector similarity search.

Upgrades TaggedCache with:
- Embedding-based semantic retrieval (replaces keyword-only matching)
- Cross-tag transferability matrix (learns which attack strategies transfer)
- Tool Generality Index (TGI) for measuring tool reusability

This is a key innovation for SCI Zone 2: from simple cache to retrieval memory.
"""

import json
import os
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict


class VectorCacheStore:
    """Retrieval-augmented attack memory with vector similarity search.

    Extends the basic tagged cache with:
    1. Semantic vector indexing for similarity-based retrieval
    2. Cross-tag transfer learning (which tools work across categories)
    3. Tool Generality Index (entropy-based diversity metric)
    """

    def __init__(
        self,
        capacity: int = 10,
        persistence_path: Optional[str] = None,
        embedding_model: Any = None,
    ):
        self.capacity = capacity
        self.persistence_path = persistence_path
        self.embedding_model = embedding_model

        # Primary cache (tag -> entries)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Vector index: {tool_id: (embedding_vector, tool_dict)}
        self._vector_index: Dict[str, Tuple[np.ndarray, Dict]] = {}

        # Cross-tag transfer matrix: {source_tag: {target_tag: success_rate}}
        self._transfer_matrix: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        if persistence_path and os.path.exists(persistence_path):
            self._load()

    # ========== Basic CRUD (extending TaggedCache) ==========

    def _ensure_tag_entry(self, tag: str):
        if tag not in self._cache:
            self._cache[tag] = {
                "successful_tools": [],
                "successful_plans": [],
                "best_approaches": [],
                "avg_score": 0.0,
                "sample_count": 0,
                "last_updated": datetime.now().isoformat(),
            }

    def add_tool(self, tag: str, tool: Any, source_tag: str = None):
        """Add a tool with optional source tag for transfer tracking."""
        self._ensure_tag_entry(tag)
        entry = self._cache[tag]
        tool_dict = tool.to_dict() if hasattr(tool, "to_dict") else tool

        existing_ids = {t.get("tool_id", "") for t in entry["successful_tools"]}
        tool_id = tool_dict.get("tool_id", "")
        if tool_id in existing_ids:
            return

        entry["successful_tools"].append(tool_dict)
        entry["successful_tools"] = sorted(
            entry["successful_tools"],
            key=lambda t: t.get("avg_judge_score", 0) or t.get("fitness", 0),
            reverse=True,
        )[:self.capacity]

        # Index for vector search
        embedding = self._compute_embedding(tool_dict)
        if embedding is not None:
            self._vector_index[tool_id] = (embedding, tool_dict)

        # Track cross-tag transfer
        if source_tag and source_tag != tag:
            score = tool_dict.get("avg_judge_score", 0) or tool_dict.get("fitness", 0)
            self._transfer_matrix[source_tag][tag].append(score)

        entry["sample_count"] += 1
        self._update_tag_stats(tag)
        self._save()

    def add_plan(self, tag: str, plan: Any):
        self._ensure_tag_entry(tag)
        entry = self._cache[tag]
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan

        existing_ids = {p.get("plan_id", "") for p in entry["successful_plans"]}
        if plan_dict.get("plan_id", "") in existing_ids:
            return

        entry["successful_plans"].append(plan_dict)
        entry["successful_plans"] = sorted(
            entry["successful_plans"],
            key=lambda p: p.get("success_count", 0) / max(1, p.get("total_attempts", 1)),
            reverse=True,
        )[:self.capacity]
        self._save()

    def add_approach(self, tag: str, approach: str):
        self._ensure_tag_entry(tag)
        entry = self._cache[tag]
        if approach not in entry["best_approaches"]:
            entry["best_approaches"].insert(0, approach)
            entry["best_approaches"] = entry["best_approaches"][:5]

    # ========== Vector Similarity Search (Key Innovation) ==========

    def search_similar(
        self, query_text: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> List[Dict]:
        """Retrieve cached tools via embedding similarity to the query.

        This replaces pure keyword-based tag matching with semantic retrieval,
        allowing cross-category knowledge transfer.
        """
        if not self._vector_index or not self.embedding_model:
            return []

        try:
            query_embedding = np.array(self.embedding_model.get_embedding(query_text))
        except Exception:
            return []

        if query_embedding.ndim == 0 or query_embedding.shape[0] == 0:
            return []

        results = []
        query_norm = np.linalg.norm(query_embedding) or 1.0

        for tool_id, (tool_emb, tool_dict) in self._vector_index.items():
            tool_norm = np.linalg.norm(tool_emb) or 1.0
            similarity = float(
                np.dot(query_embedding, tool_emb) / (query_norm * tool_norm)
            )
            if similarity >= min_similarity:
                results.append({
                    "tool_id": tool_id,
                    "similarity": similarity,
                    "tool_data": tool_dict,
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def hybrid_search(
        self, query_text: str, tag: str, top_k: int = 5, tag_weight: float = 0.3
    ) -> List[Dict]:
        """Hybrid retrieval: tag-exact matches boosted + vector similarity.

        tag_weight controls how much to boost exact-tag matches.
        """
        # Get tag-exact matches
        tag_tools = self.get_tools(tag, n=top_k)
        tag_ids = {t.get("tool_id", "") for t in tag_tools}

        # Get vector similarity matches
        vector_results = self.search_similar(query_text, top_k=top_k * 2)

        # Merge with tag boost
        merged = {}
        for t in tag_tools:
            tid = t.get("tool_id", "")
            merged[tid] = {"tool_data": t, "score": 1.0, "source": "tag_exact"}

        for vr in vector_results:
            tid = vr["tool_id"]
            boost = 1.0 + tag_weight if tid in tag_ids else 1.0
            score = vr["similarity"] * boost
            if tid not in merged or score > merged[tid]["score"]:
                merged[tid] = {
                    "tool_data": vr["tool_data"],
                    "score": score,
                    "source": "hybrid" if tid in tag_ids else "vector",
                }

        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

    # ========== Cross-Tag Transfer Learning ==========

    def get_transferable_tools(
        self, target_tag: str, min_transfer_rate: float = 0.3
    ) -> List[Dict]:
        """Find tools from other tags that transfer well to the target tag."""
        candidates = []
        for source_tag, transfers in self._transfer_matrix.items():
            if target_tag in transfers:
                rates = transfers[target_tag]
                if rates:
                    avg_rate = sum(rates) / len(rates)
                    if avg_rate >= min_transfer_rate:
                        # Get top tools from source tag
                        source_tools = self.get_tools(source_tag, n=3)
                        for t in source_tools:
                            candidates.append({
                                "tool_data": t,
                                "source_tag": source_tag,
                                "transfer_rate": avg_rate,
                                "sample_count": len(rates),
                            })

        candidates.sort(key=lambda x: x["transfer_rate"], reverse=True)
        return candidates

    def get_transfer_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the averaged cross-tag transfer matrix."""
        matrix = {}
        for src, targets in self._transfer_matrix.items():
            matrix[src] = {}
            for tgt, scores in targets.items():
                if scores:
                    matrix[src][tgt] = sum(scores) / len(scores)
        return matrix

    # ========== Tool Generality Index ==========

    def compute_tgi(self, tool_id: str) -> float:
        """Compute Tool Generality Index (entropy over tags).

        TGI = -sum(p(tag) * log(p(tag))) where p(tag) is the tool's
        success rate distribution across tags. Higher TGI = more general.
        """
        tag_scores = defaultdict(list)
        for tag, entry in self._cache.items():
            for tool in entry["successful_tools"]:
                if tool.get("tool_id", "") == tool_id:
                    score = tool.get("avg_judge_score", 0) or tool.get("fitness", 0)
                    if score > 0:
                        tag_scores[tag].append(score)

        if len(tag_scores) < 2:
            return 0.0

        # Normalize to probability distribution
        total = sum(sum(scores) for scores in tag_scores.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for tag, scores in tag_scores.items():
            p = sum(scores) / total
            if p > 0:
                entropy -= p * math.log(p)

        # Normalize by max entropy (log N)
        max_entropy = math.log(len(tag_scores))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_most_general_tools(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get tools ranked by generality across query categories."""
        tool_tgis = []
        seen = set()
        for tag, entry in self._cache.items():
            for tool in entry["successful_tools"]:
                tid = tool.get("tool_id", "")
                if tid not in seen:
                    seen.add(tid)
                    tgi = self.compute_tgi(tid)
                    tool_tgis.append((tid, tgi, tool.get("tool_name", "Unknown")))

        tool_tgis.sort(key=lambda x: x[1], reverse=True)
        return [(t[0], t[1], t[2]) for t in tool_tgis[:top_k]]

    # ========== Standard Getters (compatible with TaggedCache API) ==========

    def get_tools(self, tag: str, n: int = 3) -> List[Dict]:
        if tag not in self._cache:
            return []
        return self._cache[tag]["successful_tools"][:n]

    def get_plans(self, tag: str, n: int = 3) -> List[Dict]:
        if tag not in self._cache:
            return []
        return self._cache[tag]["successful_plans"][:n]

    def get_best_approaches(self, tag: str) -> List[str]:
        if tag not in self._cache:
            return []
        return self._cache[tag]["best_approaches"]

    def get_stats(self, tag: str) -> Dict:
        if tag not in self._cache:
            return {"avg_score": 0.0, "sample_count": 0}
        return {
            "avg_score": self._cache[tag]["avg_score"],
            "sample_count": self._cache[tag]["sample_count"],
            "tools_count": len(self._cache[tag]["successful_tools"]),
            "plans_count": len(self._cache[tag]["successful_plans"]),
        }

    def has_tag(self, tag: str) -> bool:
        return tag in self._cache and self._cache[tag]["sample_count"] > 0

    def get_all_tags(self) -> List[str]:
        return sorted(self._cache.keys())

    def get_cache_size(self) -> int:
        return len(self._vector_index)

    # ========== Internal Helpers ==========

    def _compute_embedding(self, tool_dict: Dict) -> Optional[np.ndarray]:
        if not self.embedding_model:
            return None
        text = (
            f"{tool_dict.get('tool_name', '')} "
            f"{tool_dict.get('tool_description', '')} "
            f"{tool_dict.get('approach', '')} "
            f"{tool_dict.get('tool_code', '')[:500]}"
        )
        try:
            emb = self.embedding_model.get_embedding(text)
            return np.array(emb)
        except Exception:
            return None

    def _update_tag_stats(self, tag: str):
        entry = self._cache[tag]
        tools = entry["successful_tools"]
        if tools:
            scores = [t.get("avg_judge_score", 0) or t.get("fitness", 0) for t in tools]
            entry["avg_score"] = sum(scores) / len(scores)
        entry["last_updated"] = datetime.now().isoformat()

    def _save(self):
        if not self.persistence_path:
            return
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        # Save cache (without vector index, which is rebuilt on load)
        save_data = {
            "cache": self._cache,
            "transfer_matrix": {
                src: {tgt: scores for tgt, scores in targets.items()}
                for src, targets in self._transfer_matrix.items()
            },
        }
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    def _load(self):
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cache = data.get("cache", {})
            tm = data.get("transfer_matrix", {})
            self._transfer_matrix = defaultdict(lambda: defaultdict(list))
            for src, targets in tm.items():
                for tgt, scores in targets.items():
                    self._transfer_matrix[src][tgt] = scores
        except (json.JSONDecodeError, FileNotFoundError):
            self._cache = {}

    def to_dict(self) -> Dict:
        return {
            "cache": self._cache,
            "transfer_matrix": self.get_transfer_matrix(),
            "most_general_tools": self.get_most_general_tools(5),
            "cache_size": self.get_cache_size(),
        }

    def clear(self):
        self._cache = {}
        self._vector_index = {}
        self._transfer_matrix = defaultdict(lambda: defaultdict(list))
        if self.persistence_path and os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)
