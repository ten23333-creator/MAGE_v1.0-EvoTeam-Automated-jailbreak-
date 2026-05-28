"""
Structured runtime logger for EvoTeam.

Dual-output:
- Console: respects config.verbose for filtering verbosity
- File: JSON Lines format (.jsonl) at DEBUG level, capturing everything

Usage:
    logger = EvoTeamLogger(config, session_id)
    logger.info("session_start", query="...", tag="violence")
    with logger.phase_timer("fingerprinting"):
        ...
    logger.log_llm_call("Qwen2.5-32B", "query", 1.23, 512, True)
"""

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class EvoTeamLogger:
    """Structured logger with console + JSON-lines file output."""

    def __init__(
        self,
        config: Any = None,
        session_id: str = "",
        log_dir: str = None,
        start_time: float = None,
    ):
        self.session_id = session_id
        self._start_time = start_time or time.perf_counter()

        # Resolve log directory
        logs_dir = log_dir or (config.logs_dir if config else "./attack_sessions")
        os.makedirs(logs_dir, exist_ok=True)
        self._log_path = os.path.join(logs_dir, "session.log.jsonl")

        # Read config
        verbose = getattr(config, "verbose", True)
        log_llm = getattr(config, "log_llm_calls", True) if config else True
        log_pop = getattr(config, "log_population_snapshots", True) if config else True
        log_all_conv = getattr(config, "log_all_conversations", True) if config else True

        self._log_llm_calls = log_llm
        self._log_population_snapshots = log_pop
        self._log_all_conversations = log_all_conv

        # Console handler
        console_level = logging.INFO if verbose else logging.WARNING
        self._console = logging.StreamHandler(sys.stdout)
        self._console.setLevel(console_level)
        self._console.setFormatter(logging.Formatter(
            "%(message)s"
        ))

        # File handler (DEBUG)
        self._file = logging.FileHandler(self._log_path, encoding="utf-8")
        self._file.setLevel(logging.DEBUG)
        self._file.setFormatter(logging.Formatter(
            "%(message)s"
        ))

        # Internal logger
        self._log = logging.getLogger(f"evoteam.{session_id[:8]}")
        self._log.setLevel(logging.DEBUG)
        self._log.handlers.clear()
        self._log.addHandler(self._console)
        self._log.addHandler(self._file)
        self._log.propagate = False

        # Current state
        self._iteration: Optional[int] = None
        self._phase: Optional[str] = None

        # Phase timer stack
        self._phase_start: float = 0.0

    # ---- Public API ----

    def set_iteration(self, n: int):
        self._iteration = n

    def set_phase(self, phase: str):
        self._phase = phase

    def debug(self, event: str, **data):
        self._emit("DEBUG", event, data)

    def info(self, event: str, **data):
        self._emit("INFO", event, data)

    def warning(self, event: str, **data):
        self._emit("WARNING", event, data)

    def error(self, event: str, **data):
        self._emit("ERROR", event, data)

    def sep(self, char: str = "=", width: int = 60, level: str = "DEBUG"):
        """Print a visual separator line (console only, not in file)."""
        msg = char * width
        self._log.log(logging.getLevelName(level), msg)

    @contextmanager
    def phase_timer(self, phase_name: str, iteration: int = None):
        """Context manager that logs phase_start, runs body, logs phase_end with elapsed_s."""
        prev_phase = self._phase
        prev_iter = self._iteration
        self._phase = phase_name
        if iteration is not None:
            self._iteration = iteration

        t0 = time.perf_counter()
        self.info("phase_start", phase=phase_name)
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - t0
            self.error("phase_failed", phase=phase_name, elapsed_s=round(elapsed, 4))
            raise
        else:
            elapsed = time.perf_counter() - t0
            self.info("phase_end", phase=phase_name, elapsed_s=round(elapsed, 4))
        finally:
            self._phase = prev_phase
            self._iteration = prev_iter

    def log_llm_call(
        self,
        model_name: str,
        call_type: str,
        latency_s: float,
        response_length: int,
        success: bool,
        error: str = None,
    ):
        """Record a single LLM API call."""
        if not self._log_llm_calls:
            return
        self.debug("llm_call", **{
            "model": model_name,
            "call_type": call_type,
            "latency_s": round(latency_s, 4),
            "response_length": response_length,
            "success": success,
            "error": error,
        })

    def log_population_snapshot(
        self,
        generation: int,
        tools: List[Any],
        diversity_stats: Optional[Dict] = None,
    ):
        """Record GA population state for a generation."""
        if not self._log_population_snapshots:
            return
        tool_summaries = []
        for t in tools:
            tool_summaries.append({
                "tool_name": getattr(t, "tool_name", "?"),
                "tool_id": getattr(t, "tool_id", "?"),
                "fitness": round(getattr(t, "fitness", 0.0), 4),
                "avg_judge_score": round(getattr(t.performance, "avg_judge_score", 0.0), 2),
                "generation": getattr(t, "generation", 0),
                "diversity_penalty": getattr(t, "diversity_penalty", 0.0),
                "approach": getattr(t, "approach", "")[:80],
            })
        self.info("population_snapshot", **{
            "generation": generation,
            "size": len(tool_summaries),
            "tools": tool_summaries,
            "diversity": diversity_stats or {},
        })

    def close(self):
        """Flush and close handlers."""
        for h in self._log.handlers:
            h.flush()
            h.close()
        self._log.handlers.clear()

    # ---- Internal ----

    def _emit(self, level: str, event: str, data: dict):
        """Build structured record and emit to both outputs."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "session_id": self.session_id,
            "iteration": self._iteration,
            "phase": self._phase,
            "data": data,
            "elapsed_s": round(time.perf_counter() - self._start_time, 4),
        }

        # JSON line for file
        json_line = json.dumps(record, ensure_ascii=False, default=str)

        # Console message (nice format)
        console_msg = self._format_console(level, event, data)

        log_level = getattr(logging, level)
        # File always gets the JSON
        self._file.emit(logging.LogRecord(
            "evoteam", log_level, "", 0, json_line, (), None
        ))
        # Console gets the readable message
        self._console.emit(logging.LogRecord(
            "evoteam", log_level, "", 0, console_msg, (), None
        ))

    def _format_console(self, level: str, event: str, data: dict) -> str:
        """Format a human-readable console message."""
        ts = datetime.now().strftime("%H:%M:%S")

        # Phase transitions get separator
        if event in ("phase_start",):
            phase = data.get("phase", "")
            return f"\n{ts} ── {phase.replace('_', ' ').title()} ──"

        if event == "phase_end":
            phase = data.get("phase", "")
            elapsed = data.get("elapsed_s", 0)
            return f"{ts} ── {phase.replace('_', ' ').title()} done ({elapsed:.1f}s) ──"

        # Session boundaries
        if event == "session_start":
            return f"\n{'='*50}\n{ts}  SESSION START\n{'='*50}"
        if event == "session_end":
            return f"\n{'='*50}\n{ts}  SESSION END  best={data.get('best_score','?')}  success={data.get('final_success','?')}\n{'='*50}"

        # Iteration
        if event == "iteration_start":
            return f"\n{ts} ══ Iteration {data.get('iteration','?')}/{data.get('max_iterations','?')} ══"
        if event == "iteration_end":
            return f"{ts}  Iteration {data.get('iteration','?')} end  best_score={data.get('best_score','?')}"

        # Key events
        if event == "attack_turn":
            return f"{ts}  [{data.get('tool_name','')}] turn={data.get('turn','?')}  score={data.get('score','?')}  {data.get('target_behavior','')}"
        if event == "attack_end":
            return f"{ts}  [{data.get('tool_name','')}] done  highest={data.get('highest_score','?')}  turns={data.get('total_turns','?')}  success={data.get('success','?')}"
        if event == "tool_created":
            return f"{ts}  + Tool: {data.get('tool_name','')}  gen={data.get('generation','?')}"
        if event == "population_evolved":
            return f"{ts}  GA gen={data.get('generation','?')}  size={data.get('size','?')}  avg_fitness={data.get('avg_fitness','?')}  max_fitness={data.get('max_fitness','?')}"

        # Errors/warnings
        if level in ("ERROR", "WARNING"):
            msg = data.get("message", data.get("error", ""))
            return f"{ts}  [{level}] {event}  {msg}"

        # Default: compact key=value style
        parts = [f"{k}={str(v)[:60]}" for k, v in list(data.items())[:4]]
        detail = "  ".join(parts)
        return f"{ts}  {event}  {detail}" if detail else f"{ts}  {event}"
