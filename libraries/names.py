import numpy as np
from enum import Enum


point_dtype = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32)
])

point_dtype_ext = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32),
    ('z_ref', np.float64)
])


class StixelClass(Enum):
    OBJECT: int = 0
    TOP: int = 1
