"""Base Agent Class

Provides the foundation for all specialized agents in the system.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from app.core.config import settings


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0
    ):
        """Initialize base agent

        Args:
            model: LLM model to use
            temperature: Temperature for LLM generation
        """
        self.llm = ChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature
        )
        self.agent_name = self.__class__.__name__

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main task

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary
        """
        pass

    def log(self, message: str, level: str = "INFO"):
        """Log agent activity

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR)
        """
        print(f"[{self.agent_name}] [{level}] {message}")
