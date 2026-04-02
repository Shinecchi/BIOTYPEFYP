"""
interpolation.py
----------------
Standardization Layer — Phase 2b.

Resizes a variable-length feature matrix to a fixed length using
1D Linear Interpolation (scipy.interpolate.interp1d).

Matches FYP Proposal Phase 2:
  'linear interpolation is used to normalize each window to a uniform
   length sequence... mapping sequence of variable length features
   onto an equal scale.'

Input  : List of [Dwell, Flight]  — shape (N, 2),  N is variable
Output : np.ndarray               — shape (target_length, 2),  e.g. (10, 2)
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import List


class LinearInterpolator:
    """
    Resizes any (N, 2) feature list to a fixed (target_length, 2) array.
    """

    def __init__(self, target_length: int = 10):
        if target_length < 2:
            raise ValueError("target_length must be >= 2")
        self.target_length = target_length

    def process(self, features: List[List[float]]) -> np.ndarray:
        """
        Args:
            features : List of [dwell_s, flight_s] — variable length.

        Returns:
            np.ndarray of shape (target_length, 2).
            Returns zeros if input is empty or has only 1 row.
        """
        if len(features) == 0:
            return np.zeros((self.target_length, 2))

        data = np.array(features, dtype=np.float32)  # shape (N, 2)
        n = len(data)

        if n == 1:
            # Can't interpolate a single point — tile it
            return np.tile(data, (self.target_length, 1))

        # Source x-axis: evenly spaced from 0 to n-1
        x_src = np.linspace(0, n - 1, num=n)
        # Target x-axis: evenly spaced over the same range
        x_tgt = np.linspace(0, n - 1, num=self.target_length)

        # Interpolate both columns (axis=0 = along the time dimension)
        interpolator = interp1d(x_src, data, kind='linear', axis=0)
        return interpolator(x_tgt).astype(np.float32)
