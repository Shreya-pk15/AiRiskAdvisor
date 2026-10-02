"""
Base Abstract Provider for LLM Integrations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM providers."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 30):
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output from prompt adhering to a Pydantic response_schema.

        Args:
            prompt (str): The context and instruction prompt.
            response_schema (Type[BaseModel]): Pydantic schema model to enforce response structure.
            system_instruction (Optional[str]): Optional system prompt/instruction.
            timeout (Optional[int]): Execution timeout in seconds.

        Returns:
            Dict[str, Any]: Parsed structured output matching the schema.
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Generate raw text response from prompt.

        Args:
            prompt (str): User prompt.
            system_instruction (Optional[str]): Optional system instruction.
            timeout (Optional[int]): Execution timeout in seconds.

        Returns:
            str: Generated text content.
        """
        pass
