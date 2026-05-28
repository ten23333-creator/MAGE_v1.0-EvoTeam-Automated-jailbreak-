"""
OpenAI-compatible local model client.
Supports vLLM and SGLang backends via OpenAI API protocol.
"""

import time
from typing import Optional, List, Dict, Any
from openai import OpenAI


class LocalModelError(Exception):
    """Raised when a LocalModel API call fails (only when raise_on_error=True)."""
    def __init__(self, model_name: str, method: str, original_error: Exception):
        self.model_name = model_name
        self.method = method
        self.original_error = original_error
        super().__init__(f"[{model_name}] {method} failed: {original_error}")


class LocalModel:
    """Wrapper for locally-hosted LLMs via OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_message: str = "You are a helpful assistant.",
        timeout: float = 300.0,
        logger: Any = None,
        raise_on_error: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.logger = logger
        self.raise_on_error = raise_on_error

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )

        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": system_message}
        ]

    def _track_call(self, call_type: str, latency_s: float,
                    response_length: int, success: bool, error: str = None):
        """Log an LLM API call if logger is available."""
        if self.logger and hasattr(self.logger, 'log_llm_call'):
            self.logger.log_llm_call(
                model_name=self.model_name,
                call_type=call_type,
                latency_s=latency_s,
                response_length=response_length,
                success=success,
                error=error,
            )

    def _handle_error(self, method: str, error: Exception) -> str:
        """Handle API error: log, optionally raise, or return error string."""
        error_str = str(error)
        if self.logger:
            self.logger.warning("llm_call_failed",
                model=self.model_name, method=method, error=error_str)
        if self.raise_on_error:
            raise LocalModelError(self.model_name, method, error)
        print(f"[Model Error] {self.model_name}: {error_str}")
        return f"[Model Error] {self.model_name}: {error_str}"

    def query(
        self,
        text_input: str,
        maintain_history: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a query and return the response."""
        if maintain_history:
            self.conversation_history.append({"role": "user", "content": text_input})
            messages = self.conversation_history
        else:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": text_input},
            ]

        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            response_text = response.choices[0].message.content or ""
            elapsed = time.perf_counter() - t0

            self._track_call("query", elapsed, len(response_text), True)

            if maintain_history:
                self.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )

            return response_text
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self._track_call("query", elapsed, 0, False, str(e))
            return self._handle_error("query", e)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a structured chat messages list and return the response."""
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            response_text = response.choices[0].message.content or ""
            elapsed = time.perf_counter() - t0

            self._track_call("chat", elapsed, len(response_text), True)
            return response_text
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self._track_call("chat", elapsed, 0, False, str(e))
            return self._handle_error("chat", e)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for the given text."""
        t0 = time.perf_counter()
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text.replace("\n", " "),
            )
            result = response.data[0].embedding
            elapsed = time.perf_counter() - t0
            self._track_call("embedding", elapsed, len(result), True)
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self._track_call("embedding", elapsed, 0, False, str(e))
            if self.raise_on_error:
                raise LocalModelError(self.model_name, "embedding", e)
            print(f"Embedding error: {e}")
            return []

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]

    def set_system_message(self, message: str):
        """Set a new system message and reset history."""
        self.system_message = message
        self.reset_conversation()

    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history
