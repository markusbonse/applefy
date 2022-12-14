"""
All util functions needed to estimate apertures positions
"""

import numpy as np
import math
from itertools import filterfalse


def get_number_of_apertures(separation,
                            psf_fwhm_radius):
    """
    Estimate the number of available aperture or independent pixel positions
    for a given separation.

    Args:
        separation: the separation from the star [pixel]
        psf_fwhm_radius: Radius [pixel]. Aperture positions are placed such
            that at least two radii are between them.

    Returns:
        the number of apertures at the given separation

    """

    return int(math.floor(math.pi * separation / psf_fwhm_radius))


def estimate_aperture_positions(separation,
                                center,
                                psf_fwhm_radius,
                                angle_offset=0.0):
    """
    Calculation of aperture or independent pixel positions ordered on a ring
    around the center

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


def estimate_reference_positions(planet_position,
                                 center,
                                 psf_fwhm_radius,
                                 angle_offset=0.0,
                                 safety_margin=0.0):
    """
    Calculation of aperture positions or independent pixels ordered on a ring
    around the center. The separation of the ring is given by the distance of
    the planet_position to the center. Apertures which are closer than
    safety_margin to the planet are ignored.
    
    Args:
        planet_position: The position of the planet. Is used to determine the
            separation and exclusion region. [tuple - (x, y) pixel]
        center: Position of the center [tuple - (x, y) pixel]
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
    test_positions = estimate_aperture_positions(
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
