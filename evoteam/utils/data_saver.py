"""
Data saver for persisting attack session results.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class DataSaver:
    """Simple disk-based persistence for session results."""

    def __init__(self, base_path: str = "./attack_sessions", logger=None):
        self.base_path = base_path
        self.logger = logger
        os.makedirs(base_path, exist_ok=True)

    def create_session_folder(self, query: str, target_model: str = "") -> str:
        """Create a folder for the current attack session."""
        safe_query = "".join(
            c for c in query[:30] if c.isalnum() or c in (" ", "-", "_")
        ).strip().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{safe_query}_{target_model}"
        folder_path = os.path.join(self.base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _saved(self, filepath: str):
        """Log saved file path."""
        if self.logger:
            self.logger.debug("file_saved", path=filepath)

    def save_json(self, data: Dict, folder: str, filename: str):
        """Save data as JSON file."""
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        self._saved(filepath)

    def save_session_results(self, session_data: Dict, folder: str):
        """Save complete session results."""
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "session_data": session_data,
        }, folder, "session_results.json")

    def save_all_conversations(self, context, folder: str):
        """Save ALL multi-turn conversations (not just score>=5)."""
        attack_history = getattr(context, 'attack_history', [])
        all_attacks = []
        for attack in attack_history:
            conv = attack.get('multi_turn_results', {})
            all_attacks.append({
                "query": attack.get('original_query', ''),
                "tool_name": attack.get('tool_name', ''),
                "tool_id": attack.get('tool_id', ''),
                "final_score": attack.get('final_judge_score', 0),
                "total_turns": attack.get('total_turns', 0),
                "conversation_history": conv.get('conversation_history', []),
                "scores": conv.get('scores', []),
                "conversation_successful": conv.get('conversation_successful', False),
                "highest_score": conv.get('highest_score', 0),
                "completed_at": conv.get('completed_at', ''),
            })
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "query": context.original_query,
            "tag": context.query_tag,
            "total_attacks": len(all_attacks),
            "successful_count": sum(1 for a in all_attacks if a['conversation_successful']),
            "all_attacks": all_attacks,
        }, folder, "all_conversations.json")

    def save_attack_history(self, context, folder: str):
        """Save successful attack history (score >= 5) for backward compatibility."""
        attack_history = getattr(context, 'attack_history', [])
        successful = []
        for attack in attack_history:
            if attack.get('final_judge_score', 0) >= 5:
                conv = attack.get('multi_turn_results', {})
                successful.append({
                    "query": attack.get('original_query', ''),
                    "tool_name": attack.get('tool_name', ''),
                    "final_score": attack.get('final_judge_score', 0),
                    "total_turns": attack.get('total_turns', 0),
                    "conversation_history": conv.get('conversation_history', []),
                    "conversation_successful": conv.get('conversation_successful', False),
                    "highest_score": conv.get('highest_score', 0),
                    "timestamp": conv.get('completed_at', ''),
                })
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "query": context.original_query,
            "total_successful_attacks": len(successful),
            "successful_attacks": successful,
        }, folder, "successful_attacks.json")

    def save_reflections(self, reflections: List[Dict], folder: str):
        """Persist LLM reflection outputs."""
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "reflections": reflections,
        }, folder, "reflections.json")

    def save_fingerprint(self, fingerprint: Dict, folder: str):
        """Persist full fingerprint probe details."""
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "fingerprint": fingerprint,
        }, folder, "fingerprint.json")

    def save_population_snapshot(self, generation: int, tools: List[Any],
                                  diversity_stats: Dict, folder: str):
        """Persist per-generation GA population state."""
        tools_data = []
        for tool in tools:
            if hasattr(tool, 'to_dict'):
                tools_data.append(tool.to_dict())
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "population_size": len(tools_data),
            "tools": tools_data,
            "diversity": diversity_stats,
        }, folder, f"population_gen_{generation:03d}.json")

    def save_tool_population(self, tools: List[Any], folder: str):
        """Save GA tool population state."""
        tools_data = []
        for tool in tools:
            if hasattr(tool, 'to_dict'):
                tools_data.append(tool.to_dict())
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "population_size": len(tools_data),
            "tools": tools_data,
        }, folder, "tool_population.json")

    def save_summary(self, context, folder: str):
        """Save a human-readable summary."""
        filepath = os.path.join(folder, "summary.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"EvoTeam Attack Session Summary\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Query: {context.original_query}\n")
            f.write(f"Tag: {context.query_tag}\n")
            f.write(f"Total Attacks: {context.total_attacks}\n")
            f.write(f"Successful: {context.successful_attacks}\n")
            f.write(f"Success Rate: {context.success_rate:.1%}\n")
            f.write(f"Total Tools: {len(context.created_tools)}\n")
            f.write(f"Population Size: {len(context.tool_population)}\n")
            f.write(f"\nTop Tools:\n")
            for i, tool in enumerate(context.get_best_tools(5), 1):
                f.write(f"  {i}. {tool.tool_name} (fitness={tool.fitness:.3f}, "
                        f"avg_score={tool.performance.avg_judge_score:.1f})\n")
        self._saved(filepath)