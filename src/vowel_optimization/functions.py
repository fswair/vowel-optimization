"""Function cases for optimization.

Each case is a function with known implementation (ground truth) and a description.
GEPA will optimize the eval generation prompt so that generated evals
correctly validate these original implementations.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from .definitions import FUNCTIONS


@dataclass
class FunctionCase:
    """A single function case for optimization."""

    name: str
    description: str
    func: Callable
    code: str


def get_function_cases() -> list[FunctionCase]:
    """Build FunctionCase list from playground originals."""
    cases = []
    for name, info in FUNCTIONS.items():
        cases.append(
            FunctionCase(
                name=name,
                description=info["description"],
                func=info["func"],
                code=inspect.getsource(info["func"]),
            )
        )
    return cases


# Pre-built list for convenience
FUNCTION_CASES = get_function_cases()
