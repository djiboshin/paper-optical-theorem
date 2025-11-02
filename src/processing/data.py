from dataclasses import dataclass
import numpy as np
import torch
from typing import TypeVar, Generic


DataType = TypeVar("DataType", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class DataContainer(Generic[DataType]):
    """Immutable data container for storing pressure fields, corresponding z positions, and frequencies."""

    p: DataType
    z: DataType
    freq: DataType
