"""
General functions needed in multiple utils
"""

import numpy as np
from typing import Tuple


def center_subpixel(image: np.ndarray) -> Tuple[float, float]:
    """
    Code copied from pynpoint

    Function to get the precise position of the image center. The center of the
    pixel in the bottom left corner of the image is defined as (0, 0), so the
    bottom left corner of the image is located at (-0.5, -0.5).

    Args:
        image : np.ndarray
            Input image (2D or 3D).

    Returns:
        tuple(float, float)
        Subpixel position (y, x) of the image center.
    """

    center_x = float(image.shape[-1]) / 2 - 0.5
    center_y = float(image.shape[-2]) / 2 - 0.5

    return center_x, center_y
