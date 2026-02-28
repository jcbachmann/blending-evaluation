"""
Blending Simulator Lib for Python
"""
from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing

__all__: list[str] = ['BlendingSimulatorLib']


class BlendingSimulatorLib:
    def __init__(self, arg0: typing.SupportsFloat | typing.SupportsIndex,
                 arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex,
                 arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: bool,
                 arg5: typing.SupportsFloat | typing.SupportsIndex, arg6: typing.SupportsFloat | typing.SupportsIndex,
                 arg7: typing.SupportsFloat | typing.SupportsIndex, arg8: bool,
                 arg9: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...

    def get_heights(self) -> list[list[float]]:
        ...

    def reclaim(self) -> dict:
        ...

    def stack(self, arg0: typing.SupportsFloat | typing.SupportsIndex,
              arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex,
              arg3: typing.SupportsFloat | typing.SupportsIndex,
              arg4: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...

    def stack_list(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
                   arg1: collections.abc.Sequence[str]) -> None:
        ...
