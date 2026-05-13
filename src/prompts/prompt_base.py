"""Base classes and interfaces for prompt construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from core.schemas import QueryInstance, TaskSpec


@dataclass
class PromptContext:
	"""Context for prompt generation."""

	task_spec: TaskSpec
	query: QueryInstance
	retrieved_sets: dict[int, Any] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptOutput:
	"""Structured output from prompt generation."""

	system_prompt: str
	user_message: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_messages(self) -> list[dict]:
		"""Convert to OpenAI message format."""
		messages = []
		if self.system_prompt:
			messages.append({"role": "system", "content": self.system_prompt})
		messages.append({"role": "user", "content": self.user_message})
		return messages


class PromptBuilder(ABC):
	"""
	Abstract base class for task-specific prompt builders.

	Each task type (classification, prediction, anomaly) has its own builder
	that knows how to format data and construct appropriate prompts.
	"""

	def __init__(
		self,
		name: Optional[str] = None,
		config: Optional[dict[str, Any]] = None,
	) -> None:
		self.name = name or self.__class__.__name__
		self.config = config or {}

	@abstractmethod
	def build_system_prompt(self, context: PromptContext) -> str:
		"""
		Build the system prompt for the given task.

		System prompts define the role and behavior of the LLM.
		"""
		...

	@abstractmethod
	def build_user_message(self, context: PromptContext) -> str:
		"""
		Build the user message containing the query and retrieved examples.

		User messages contain the specific task instance to solve.
		"""
		...

	def build(self, context: PromptContext) -> PromptOutput:
		"""Build complete prompt output."""
		system_prompt = self.build_system_prompt(context)
		user_message = self.build_user_message(context)
		return PromptOutput(
			system_prompt=system_prompt,
			user_message=user_message,
			metadata={"builder": self.name, "task_type": context.task_spec.task_type},
		)

	def get_config(self, key: str, default: Any = None) -> Any:
		"""Get configuration value."""
		return self.config.get(key, default)
