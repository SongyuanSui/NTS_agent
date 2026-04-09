# src/core/exceptions.py

from __future__ import annotations


class TSAgentError(Exception):
    """
    Base exception for the project.
    """


class ConfigError(TSAgentError):
    """
    Raised when configuration is invalid or incomplete.
    """


class RegistryError(TSAgentError):
    """
    Raised when registry lookup / registration fails.
    """


class FactoryError(TSAgentError):
    """
    Raised when object construction from config fails.
    """


class ValidationError(TSAgentError):
    """
    Raised when runtime inputs / schemas are invalid.
    """


class PipelineError(TSAgentError):
    """
    Raised when pipeline execution fails in a structured way.
    """


class AgentExecutionError(TSAgentError):
    """
    Raised when an agent fails during execution.
    """


class ArtifactError(TSAgentError):
    """
    Raised when artifact load/save/resolve fails.
    """