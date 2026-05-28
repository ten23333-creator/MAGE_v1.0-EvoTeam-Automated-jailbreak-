"""
Restricted sandbox executor for AI-generated attack tool code.

Provides a secure execution environment with:
- Restricted builtins (no __import__, exec, eval, open, etc.)
- Timeout enforcement (cross-platform via daemon thread + join timeout)
- Allowed imports whitelist
- Execution result capture
"""

import sys
import io
import traceback
import threading
from typing import Any, Dict, Optional, Set


# Dangerous builtins that should be restricted
RESTRICTED_BUILTINS = {
    "__import__", "exec", "eval", "compile", "open", "input",
    "breakpoint", "exit", "quit", "help", "copyright", "license",
    "credits", "memoryview",
}

# Safe builtins to allow
SAFE_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "complex", "delattr", "dict",
    "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "globals", "hasattr", "hash", "hex", "id", "int",
    "isinstance", "issubclass", "iter", "len", "list", "locals", "map",
    "max", "min", "next", "object", "oct", "ord", "pow", "print",
    "property", "range", "repr", "reversed", "round", "set", "setattr",
    "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple",
    "type", "vars", "zip", "True", "False", "None", "Exception",
    "ValueError", "TypeError", "KeyError", "IndexError", "StopIteration",
    "RuntimeError", "ArithmeticError", "OSError", "IOError", "ZeroDivisionError",
}


class SandboxExecutor:
    """Restricted execution environment for AI-generated tool code."""

    def __init__(
        self,
        timeout: int = 30,
        allowed_imports: Optional[Set[str]] = None,
        max_code_length: int = 200,
    ):
        self.timeout = timeout
        self.allowed_imports = allowed_imports or {
            "json", "re", "random", "string", "textwrap",
            "itertools", "collections", "math", "functools",
        }
        self.max_code_length = max_code_length

    def execute(self, tool_code: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool code in a restricted environment with timeout enforcement.

        Uses daemon thread + join(timeout) for reliable cross-platform timeout.
        If the thread outlives the timeout it runs on (orphan daemon) but the
        main process returns the timeout error immediately.
        """
        # Validate code length
        lines = tool_code.strip().split("\n")
        if len(lines) > self.max_code_length:
            return {
                "success": False,
                "result": None,
                "error": f"Code too long: {len(lines)} lines (max {self.max_code_length})",
                "stdout": "",
                "stderr": "",
                "sandbox_violation": "code_length_exceeded",
            }

        # Shared container so the thread can pass its result back
        result_holder = {}

        def _target():
            try:
                result_holder["data"] = self._execute_inner(
                    tool_code, query, context
                )
            except Exception as exc:
                result_holder["exception"] = exc

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            # Thread still running => code hit infinite loop or stalled
            return {
                "success": False,
                "result": None,
                "error": f"Execution timeout after {self.timeout}s",
                "stdout": "",
                "stderr": "",
                "sandbox_violation": "timeout",
            }

        if "exception" in result_holder:
            raise result_holder["exception"]

        return result_holder.get("data", {
            "success": False,
            "result": None,
            "error": "Thread returned no result",
            "stdout": "",
            "stderr": "",
        })

    def _execute_inner(
        self, tool_code: str, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Core execution logic (runs inside a thread for timeout wrapping)."""
        # Prepare restricted globals
        restricted_globals = {
            "__builtins__": {
                k: __builtins__[k] if k in __builtins__ else getattr(__builtins__, k, None)
                for k in SAFE_BUILTINS
            },
            "__name__": "__sandbox__",
            "__doc__": None,
            "query": query,
            "context": context,
        }

        # Add allowed imports
        restricted_globals["__builtins__"]["__import__"] = self._make_safe_import()

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Compile and execute
            compiled = compile(tool_code, "<sandbox_tool>", "exec")
            exec(compiled, restricted_globals)

            # Try to get the result
            result = None
            if "execute" in restricted_globals:
                result = restricted_globals["execute"](query, context)
            elif "main" in restricted_globals:
                result = restricted_globals["main"](query, context)
            elif "attack" in restricted_globals:
                result = restricted_globals["attack"](query, context)
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": "No execute/main/attack function found in tool code",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "sandbox_violation": "missing_entry_point",
                }

            return {
                "success": True,
                "result": result,
                "error": None,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "sandbox_violation": None,
            }

        except SyntaxError as e:
            return {
                "success": False,
                "result": None,
                "error": f"SyntaxError: {str(e)}",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "sandbox_violation": None,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "sandbox_violation": None,
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _make_safe_import(self):
        """Create a restricted __import__ function."""
        allowed = self.allowed_imports

        def safe_import(name, *args, **kwargs):
            # Only allow whitelisted modules
            if name not in allowed and not any(
                name.startswith(f"{a}.") for a in allowed
            ):
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
            return __import__(name, *args, **kwargs)

        return safe_import

    def validate_code_structure(self, tool_code: str) -> Dict[str, Any]:
        """Perform static analysis on tool code before execution."""
        issues = []

        # Check for dangerous patterns
        dangerous_patterns = [
            ("import os", "os module import"),
            ("import subprocess", "subprocess import"),
            ("import sys", "sys module import - restricted"),
            ("import shutil", "shutil import"),
            ("import socket", "socket import"),
            ("import requests", "direct requests import - use context API instead"),
            ("import urllib", "urllib import"),
            ("import pickle", "pickle import - unsafe deserialization"),
            ("os.system", "os.system call"),
            ("subprocess.", "subprocess usage"),
        ]

        for pattern, issue in dangerous_patterns:
            if pattern in tool_code:
                issues.append({"pattern": pattern, "issue": issue, "severity": "high"})

        return {
            "valid": len([i for i in issues if i["severity"] == "high"]) == 0,
            "issues": issues,
            "line_count": len(tool_code.strip().split("\n")),
        }

    def classify_safety_level(self, tool_code: str) -> Dict[str, Any]:
        """Classify the safety level of AI-generated code.

        Safety Levels:
          L0 (SAFE): Pure text/string operations, no I/O, no imports beyond stdlib
          L1 (RESTRICTED): Uses whitelisted imports, no dangerous patterns
          L2 (BOUNDARY): Contains patterns that need review (e.g., string encoding tricks)
          L3 (UNSAFE): Contains dangerous patterns (file access, network calls, etc.)

        Returns:
            Dict with safety_level, level_name, reasons, and recommended_action
        """
        code_lower = tool_code.lower()

        # Check for Level 3 patterns (clearly dangerous)
        l3_patterns = [
            "import os", "import subprocess", "import socket",
            "import requests", "import urllib", "import pickle",
            "os.system", "subprocess.", "eval(", "exec(",
            "open(", "__import__", "globals()", "locals()",
            "getattr(__builtins__", "setattr(__builtins__",
        ]
        for pattern in l3_patterns:
            if pattern in code_lower:
                return {
                    "safety_level": 3,
                    "level_name": "UNSAFE",
                    "reasons": [f"Dangerous pattern detected: {pattern}"],
                    "recommended_action": "REJECT - Do not execute",
                }

        # Check for Level 2 patterns (boundary cases)
        l2_patterns = [
            "compile(", "bytes(", "bytearray(", "memoryview(",
            "base64", "encode(", "decode(", "chr(", "ord(",
            "__getattribute__", "__setattr__", "vars(",
        ]
        l2_found = [p for p in l2_patterns if p in code_lower]
        if l2_found:
            return {
                "safety_level": 2,
                "level_name": "BOUNDARY",
                "reasons": [f"Boundary pattern: {p}" for p in l2_found],
                "recommended_action": "REVIEW - Execute with monitoring",
            }

        # Check for Level 1 patterns (restricted but allowed imports)
        imports_used = []
        for allowed_mod in self.allowed_imports:
            if f"import {allowed_mod}" in code_lower or \
               f"from {allowed_mod}" in code_lower:
                imports_used.append(allowed_mod)

        if imports_used:
            return {
                "safety_level": 1,
                "level_name": "RESTRICTED",
                "reasons": [f"Uses whitelisted imports: {imports_used}"],
                "recommended_action": "EXECUTE - Standard sandbox",
            }

        # Level 0: Pure Python, no external imports
        return {
            "safety_level": 0,
            "level_name": "SAFE",
            "reasons": ["Pure Python, no external imports"],
            "recommended_action": "EXECUTE - Minimal risk",
        }

    def get_safety_report(self, tool_code: str) -> Dict[str, Any]:
        """Get a comprehensive safety report combining static analysis and
        safety level classification."""
        static_analysis = self.validate_code_structure(tool_code)
        safety_class = self.classify_safety_level(tool_code)

        return {
            "overall_risk": safety_class["safety_level"],
            "static_analysis": static_analysis,
            "safety_classification": safety_class,
            "executable": static_analysis["valid"] and safety_class["safety_level"] < 3,
            "line_count": static_analysis["line_count"],
        }