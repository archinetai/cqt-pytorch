from typing import Optional, TypeVar

import torch
from typing_extensions import TypeGuard

T = TypeVar("T")


"""
Utils
"""


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


"""
CQT
"""

class CQT(nn.Module):

    def __init__(
        self,
    ):
        super().__init__() 


    def encode(self, x: Tensor) -> Tensor:
        pass 


    def decode(self, x: Tensor) -> Tensor:
        pass 