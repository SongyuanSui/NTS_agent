from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from core.enums import RepresentationType
from core.interfaces import BaseRepresentationInterface
from core.schemas import TimeSeriesSample
from representations.schemas import RepresentationInput, RepresentationOutput


class BaseRepresentation(BaseRepresentationInterface, ABC):
    """
    Base implementation for all representation modules.

    Implementations should transform one batch of TimeSeriesSample objects
    into a typed representation output.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)

    @property
    @abstractmethod
    def rep_type(self) -> RepresentationType:
        """Representation type produced by this component."""

    def validate_input(self, input_data: RepresentationInput) -> None:
        if not isinstance(input_data, RepresentationInput):
            raise TypeError("input_data must be a RepresentationInput.")

        if len(input_data.samples) == 0:
            raise ValueError("input_data.samples must be non-empty.")

        for sample in input_data.samples:
            if not isinstance(sample, TimeSeriesSample):
                raise TypeError("input_data.samples must contain TimeSeriesSample objects only.")

    def normalize_context(self, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if context is None:
            return {}
        if not isinstance(context, dict):
            raise TypeError("context must be a dict or None.")
        return context

    @abstractmethod
    def transform(
        self,
        input_data: RepresentationInput,
        context: Optional[dict[str, Any]] = None,
    ) -> RepresentationOutput:
        """Compute one representation batch from input samples."""

    def run(
        self,
        input_data: RepresentationInput,
        context: Optional[dict[str, Any]] = None,
    ) -> RepresentationOutput:
        context = self.normalize_context(context)
        self.validate_input(input_data)
        return self.transform(input_data=input_data, context=context)
