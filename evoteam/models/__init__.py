"""EvoTeam model interfaces."""

from .openai_model import LocalModel, LocalModelError

__all__ = ["LocalModel", "LocalModelError"]