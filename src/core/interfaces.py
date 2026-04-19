# src/core/interfaces.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from .enums import RepresentationType, TaskType
from .schemas import (
    BatchPredictionRecord,
    PipelineResult,
    QueryInstance,
    TaskSpec,
    TimeSeriesSample,
)


@runtime_checkable
class SupportsDescribe(Protocol):
    """
    Lightweight protocol for components that can expose a short description.
    """

    def describe(self) -> str:
        ...


class BaseComponent(ABC):
    """
    Minimal shared base class for major framework components.

    This class is intentionally thin. It provides:
    - a stable component name
    - optional config storage
    - a human-readable description hook

    Concrete subclasses should decide their own runtime behavior.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        self._name = name or self.__class__.__name__
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def describe(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class BaseDatasetLoaderInterface(BaseComponent, ABC):
    """
    Base interface for all dataset loaders.
    
    A dataset loader is responsible for loading raw datasets and returning
    task-specific bundles (train/test splits, labels, etc.).
    """

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Task type this loader supports."""
        raise NotImplementedError

    @abstractmethod
    def load(self, dataset_name: str, base_dir: str | Path, **kwargs: Any) -> Any:
        """Load one dataset and return a task-specific bundle."""
        raise NotImplementedError


class BaseRepresentationInterface(BaseComponent, ABC):
    """
    Base interface for all representation modules.

    Implementations should transform one batch of TimeSeriesSample objects
    into a typed representation output.
    """

    @property
    @abstractmethod
    def rep_type(self) -> RepresentationType:
        """Representation type produced by this component."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        input_data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Compute one representation batch from input samples."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        input_data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Validate input, then compute representation.
        
        This is the main entry point for representation computation.
        """
        raise NotImplementedError


class BaseTaskInterface(BaseComponent, ABC):
    """
    Base interface for all task definitions.

    A task implementation is responsible for translating a raw sample into the
    framework's internal query object, defining its label space, and parsing
    raw model/agent outputs into standardized predictions.
    """

    def __init__(self, task_spec: TaskSpec, name: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self._task_spec = task_spec

    @property
    def task_spec(self) -> TaskSpec:
        return self._task_spec

    @abstractmethod
    def build_query(self, sample: TimeSeriesSample) -> QueryInstance:
        """
        Convert one raw sample into a standardized query object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_label_space(self) -> list[str]:
        """
        Return the label space for this task.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt_target(self) -> str:
        """
        Return a short task instruction string used by prompt builders.
        Examples:
        - 'predict the class label'
        - 'predict the future label'
        - 'determine whether the window is anomalous'
        """
        raise NotImplementedError

    @abstractmethod
    def parse_output(self, raw_output: Any, sample: TimeSeriesSample) -> Any:
        """
        Parse a raw output from a downstream component into a task-aligned object.

        The return type is intentionally left flexible here because some tasks
        may first parse into intermediate objects before converting to a final
        PredictionRecord elsewhere.
        """
        raise NotImplementedError


class BaseAgentInterface(BaseComponent, ABC):
    """
    Base interface for all agents.

    Every concrete agent should:
    - validate its input
    - run one well-defined operation
    - return a structured schema object, not an arbitrary dict
    """

    @abstractmethod
    def validate_input(self, input_data: Any) -> None:
        """
        Validate input before execution.

        Implementations should raise ValueError / TypeError (or framework-
        specific exceptions) when the input is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, input_data: Any, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Execute the agent on one input object and return one structured output object.
        """
        raise NotImplementedError


class BaseRetrieverInterface(BaseComponent, ABC):
    """
    Base interface for retrieval modules.

    A retriever scores candidates from a memory bank against a query and returns
    a structured retrieval result.
    """

    @abstractmethod
    def validate_query(self, query: QueryInstance) -> None:
        """
        Validate the query object before retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, query: QueryInstance, candidate: Any) -> float:
        """
        Compute a similarity or distance-derived score between query and candidate.

        Notes
        -----
        - The exact meaning of the score depends on the retriever.
        - Higher-is-better or lower-is-better policy should be documented by the
          concrete implementation and normalized before fusion if needed.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: QueryInstance,
        memory_bank: Any,
        top_k: int,
        context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Retrieve top-k candidates from the memory bank for the given query.

        Implementations should return a structured retrieval schema object.
        """
        raise NotImplementedError


class BaseEvaluatorInterface(BaseComponent, ABC):
    """
    Base interface for all evaluators.

    An evaluator computes metrics on predictions or retrieval outputs and returns
    structured metric results.
    """

    @abstractmethod
    def validate_input(self, data: Any) -> None:
        """
        Validate input data before evaluation.

        Implementations should raise ValueError / TypeError when the input is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Evaluate the input data and return metric results.

        Returns a dict containing metric names and their values.
        """
        raise NotImplementedError


class BasePipelineInterface(BaseComponent, ABC):
    """
    Base interface for all pipelines.

    A pipeline is an orchestrator: it wires tasks, agents, and retrieval modules
    into an end-to-end execution flow.
    """

    @abstractmethod
    def validate_components(self) -> None:
        """
        Validate that required components are present and compatible.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Run the pipeline on a single sample.
        """
        raise NotImplementedError

    @abstractmethod
    def run_batch(
        self,
        samples: Iterable[TimeSeriesSample],
        context: Optional[dict[str, Any]] = None,
    ) -> BatchPredictionRecord:
        """
        Run the pipeline on multiple samples.
        """
        raise NotImplementedError


class SupportsMemoryBank(Protocol):
    """
    Optional protocol for memory-like objects used by retrieval modules.
    """

    def __len__(self) -> int:
        ...

    def get_all(self) -> list[Any]:
        ...


class SupportsRegistryLookup(Protocol):
    """
    Optional protocol for registry-like objects.
    """

    def get(self, name: str) -> Any:
        ...


class SupportsSerialization(Protocol):
    """
    Optional protocol for components or outputs that can be serialized.
    """

    def to_dict(self) -> dict[str, Any]:
        ...