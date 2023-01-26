"""
Util functions needed to estimate noise positions in residuals and raw data.
"""

from typing import Tuple, List
from itertools import filterfalse

import math
import numpy as np


def get_number_of_apertures(
        separation: float,
        psf_fwhm_radius: float
) -> int:
    """
    Estimates the number of available aperture or independent pixel positions
    for a given separation.

    Args:
        separation: the separation from the star [pixel]
        psf_fwhm_radius: Radius [pixel]. Aperture positions are placed such
            that at least two radii are between them.

    Returns:
        The number of apertures at the given separation.

    """

    return int(math.floor(math.pi * separation / psf_fwhm_radius))


def estimate_noise_positions(
        separation: float,
        center: Tuple[float, float],
        psf_fwhm_radius: float,
        angle_offset: float = 0.0
) -> List[Tuple[float, float, float, float]]:
    """
    Calculation of aperture or independent pixel positions ordered on a ring
    around the center.

    Args:
        separation: Separation of the ring from the center [pixel]
        center: Position of the center [tuple - (x, y) pixel]
        psf_fwhm_radius: Radius [pixel]. positions are placed such that at
            least two radii are between them.
        angle_offset: offset angle [rad] which is applied to rotate the
            aperture positions around the center.

    Returns:
        A list of tuples (x_pos, y_pos, separation, angle)
        containing the positions of the apertures
    """

    num_ap = get_number_of_apertures(separation=separation,
                                     psf_fwhm_radius=psf_fwhm_radius)

    ap_theta = np.linspace(0, 2 * np.pi, num_ap, endpoint=False)

    all_positions = []

    for theta in ap_theta:
        s_tmp = separation
        theta_new = theta + angle_offset
        alpha_tmp = np.rad2deg(theta_new)
        x_tmp = center[1] + np.cos(theta_new) * separation
        y_tmp = center[0] + np.sin(theta_new) * separation
        all_positions.append((x_tmp, y_tmp, s_tmp, alpha_tmp))

    return all_positions


def estimate_reference_positions(
        planet_position: Tuple[float, float],
        center: Tuple[float, float],
        psf_fwhm_radius: float,
        angle_offset: float = 0.0,
        safety_margin: float = 0.0
) -> List[Tuple[float, float, float, float]]:
    """
    Calculation of aperture positions or independent pixels ordered on a ring
    around the center. The separation of the ring is given by the distance of
    the planet_position to the center. Apertures which are closer than
    safety_margin to the planet are ignored.

    Args:
        planet_position: The position of the planet. Is used to determine the
            separation and exclusion region. [pixel]
        center: Position of the center [pixel]
        psf_fwhm_radius: Radius [pixel]. positions are placed such that at
            least two radii are between them.
        angle_offset: offset angle [rad] which is applied to rotate the
            aperture positions around the center.
        safety_margin: separation from the planet_position in which apertures
            are ignored.

    Returns:
        A list of tuples (x_pos, y_pos, separation, angle)
        containing the positions of the apertures
    """
    # Ignore the aperture of the planet
    safety_margin += 2 * psf_fwhm_radius

    # 1.) Compute distance of the planet to the center
    distance = np.linalg.norm(np.array(planet_position) -
                              np.array(center))

    # 2.) sample positions
    test_positions = estimate_noise_positions(
        distance,
        center=center,
        psf_fwhm_radius=psf_fwhm_radius,
        angle_offset=angle_offset)

    # 3.) sort out positions close to the planet
    def check_distance(tmp_pos):
        tmp_distance = np.linalg.norm(
            np.array(tmp_pos[:2]) - np.array(planet_position))
        return tmp_distance < safety_margin

    test_positions[:] = filterfalse(check_distance, test_positions)

    return test_positions


def center_subpixel(image: np.ndarray) -> Tuple[float, float]:
    """
    Code copied from `PynPoint <https://pynpoint.readthedocs.io/en/latest/>`_.

    Function to get the precise position of the image center. The center of the
    pixel in the bottom left corner of the image is defined as (0, 0), so the
    bottom left corner of the image is located at (-0.5, -0.5).

    Args:
        image : np.ndarray
            Input image (2D or 3D).

    Returns:
        Subpixel position (y, x) of the image center.
    """

    center_x = float(image.shape[-1]) / 2 - 0.5
    center_y = float(image.shape[-2]) / 2 - 0.5

    return center_x, center_y
