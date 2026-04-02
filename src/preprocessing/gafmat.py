"""
gafmat.py
---------
Feature Transformation Layer — Phase 2c.

Converts a fixed-length (10, 2) feature matrix into a (10, 10, 2)
Gramian Angular Summation Field (GASF) image tensor.

Matches FYP Proposal:
  'GAFMAT represents this one-dimensional temporal sequence as a
   two-dimensional matrix by modelling pairwise temporal relationships.'

Input  : np.ndarray of shape (target_length, 2)   — e.g. (10, 2)
Output : np.ndarray of shape (target_length, target_length, 2) — e.g. (10, 10, 2)
"""

import numpy as np
from pyts.image import GramianAngularField


class GAFMATTransformer:
    """
    Transforms a (N, 2) feature matrix into an (N, N, 2) GAFMAT image.

    Channel 0 = Dwell Time  GAF pattern
    Channel 1 = Flight Time GAF pattern
    """

    def __init__(self, image_size: int = 10):
        self.image_size = image_size
        # 'summation' = Gramian Angular Summation Field (GASF)
        # Automatically normalises data to [-1, 1] internally
        self.gaf = GramianAngularField(image_size=image_size, method='summation')

    def transform(self, standardized_features: np.ndarray) -> np.ndarray:
        """
        Args:
            standardized_features : np.ndarray of shape (target_length, 2)

        Returns:
            np.ndarray of shape (image_size, image_size, 2)

        Raises:
            ValueError if input does not have exactly 2 feature columns.
        """
        if standardized_features.ndim != 2 or standardized_features.shape[1] != 2:
            raise ValueError(
                f"Expected shape (N, 2), got {standardized_features.shape}"
            )

        # pyts expects (n_samples, n_timestamps) — so each feature is 1 sample
        dwell_row  = standardized_features[:, 0].reshape(1, -1)  # (1, N)
        flight_row = standardized_features[:, 1].reshape(1, -1)  # (1, N)

        dwell_gaf  = self.gaf.fit_transform(dwell_row)[0]   # (N, N)
        flight_gaf = self.gaf.fit_transform(flight_row)[0]  # (N, N)

        # Stack along last axis → (N, N, 2)
        return np.stack((dwell_gaf, flight_gaf), axis=-1).astype(np.float32)
