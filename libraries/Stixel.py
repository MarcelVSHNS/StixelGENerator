import numpy as np
from enum import Enum
from typing import Dict


point_dtype_sph = np.dtype([
    ('r', np.float64),
    ('az', np.float64),
    ('el', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32),
    ('sem_seg', np.int32)
])


point_dtype = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32),
    ('sem_seg', np.int32)
])

point_dtype_stx = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32),
    ('w', np.float64)
])

point_dtype_ext = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32),
    ('sem_seg', np.int32),
    ('z_ref', np.float64)
])


class StixelClass(Enum):
    OBJECT: int = 0
    TOP: int = 1


class Stixel:
    """
        Represents a stixel in an image.
        Args:
            top_point (np.array): The top point of the stixel in the image coordinate system.
            bottom_point (np.array): The bottom point of the stixel in the image coordinate system.
            position_class (StixelClass): The position class of the stixel.
            image_size (Dict[str, int]): The size of the image in pixels (width and height).
            grid_step (int, optional): The step size of the grid. Defaults to 8.
        Attributes:
            column (int): The column (x-coordinate) of the stixel in the image grid.
            top_row (int): The top row (y-coordinate) of the stixel in the image grid.
            bottom_row (int): The bottom row (y-coordinate) of the stixel in the image grid.
            position_class (StixelClass): The position class of the stixel.
            top_point (np.array): The top point of the stixel in the image coordinate system.
            bottom_point (np.array): The bottom point of the stixel in the image coordinate system.
            depth (float): The depth of the stixel.
            image_size (Dict[str, int]): The size of the image in pixels (width and height).
            grid_step (int): The step size of the grid.
        Raises:
            AssertionError: If the stixel is not within the image bounds or does not align with the grid.
    """
    def __init__(self,
                 top_point: np.array,
                 bottom_point: np.array,
                 position_class: StixelClass,
                 image_size: Dict[str, int],
                 grid_step: int = 8):
        self.column = top_point['proj_x']
        self.top_row = top_point['proj_y']
        self.bottom_row = bottom_point['proj_y']
        self.position_class: StixelClass = position_class
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.depth = self.calculate_depth(top_point)
        self.image_size = image_size
        self.grid_step = grid_step
        self.sem_seg = top_point['sem_seg']

        self.force_stixel_to_grid()
        self.check_integrity()

    def force_stixel_to_grid(self):
        """
        Force the stixel position to align with the grid.
        This method adjusts the stixel position to align with the grid by normalizing the top and bottom row
        coordinates, and the column coordinate. If the normalized row exceeds the image height, it is adjusted to be
        the maximum possible position within the grid. If the top row and bottom row coincide, the top row position is
        adjusted accordingly. Finally, the column coordinate is normalized, and if it equals the image width, it is
        adjusted to be the maximum possible position within the grid.
        """
        for attr in ('top_row', 'bottom_row'):
            normalized_row = self._normalize_into_grid(getattr(self, attr), step=self.grid_step)
            if normalized_row >= self.image_size['height']:
                normalized_row = self.image_size['height'] - self.grid_step
            setattr(self, attr, normalized_row)
        if self.top_row == self.bottom_row:
            if self.top_row == self.image_size['height'] - self.grid_step:
                self.top_row -= self.grid_step
            else:
                self.bottom_row += self.grid_step
        self.column = self._normalize_into_grid(self.column, step=self.grid_step)
        if self.column >= self.image_size['width']:
            self.column = self.image_size['width'] - self.grid_step

    def check_integrity(self):
        """
        Check the integrity of the values for the specified column and row.
        """
        for cut_row in (self.top_row, self.bottom_row):
            assert cut_row <= self.image_size['height'], f"y-value out of bounds ({self.column},{cut_row})."
            assert cut_row % self.grid_step == 0, f"y-value is not into grid ({self.column},{cut_row})."
        assert self.column <= self.image_size['width'], f"x-value out of bounds ({self.column},{cut_row})."
        assert self.column % self.grid_step == 0, f"x-value is not into grid ({self.column},{cut_row})."
        #assert self.top_row < self.bottom_row, f"Top is higher than Bottom. Top_pt: {self.top_point}. Bottom_pt:{self.bottom_point}."
        assert self.top_row != self.bottom_row, "Top is Bottom."

    @staticmethod
    def _normalize_into_grid(pos: int, step: int = 8):
        """
        Args:
            pos (int): The position value to be normalized into the grid.
            step (int, optional): The grid step size. Defaults to 8.
        Returns:
            int: The normalized value of pos, rounded down to the nearest multiple of step.
        """
        # Calculate the remainders from rounding down and up
        remainder_down = pos % step
        remainder_up = step - remainder_down

        # Check if the remainder from rounding up is smaller (closer)
        if remainder_up < remainder_down:
            val_norm = pos + remainder_up  # Round up
        else:
            val_norm = pos - remainder_down  # Round down

        return val_norm

    @staticmethod
    def calculate_depth(top_point):
        """
        Calculate the depth from a given top point.
        Args:
            top_point (dict): The coordinates of the top point.
                The coordinates should be specified as the 'x', 'y', and 'z' keys in a dictionary.
        Returns:
            float: The depth calculated as the Euclidean distance from the origin to the top point.
        """
        depth = np.sqrt(top_point['x'] ** 2 + top_point['y'] ** 2 + top_point['z'] ** 2)
        return depth
