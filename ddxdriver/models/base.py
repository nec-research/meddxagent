from abc import ABC, abstractmethod
from typing import Dict, List


class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """Setup LLM configs here"""
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        message_history: List[Dict[str, str]] | None = None,
        max_tokens: int = None,
        temperature=0.0,
        **kwargs,
    ) -> str:
        """
        Prompt the LLM and get the response text
        If predefined_messages is not None or empty, then it will use the user and system prompts provided
        """
        raise NotImplementedError
