"""
Functions which can be used to estimate the uncertainty (p-value) that a
potential planet at a given position in a residual is not noise.
"""

from typing import Union, Tuple

import numpy as np

from applefy.statistics.general import TestInterface
from applefy.utils.photometry import AperturePhotometryMode, get_flux, \
    IterNoiseForPlanet


def compute_detection_uncertainty(
        frame: np.ndarray,
        planet_position: Union[Tuple[float, float],
                               Tuple[float, float, float, float]],
        statistical_test: TestInterface,
        psf_fwhm_radius: float,
        photometry_mode_planet: AperturePhotometryMode,
        photometry_mode_noise: AperturePhotometryMode,
        safety_margin: float = 1.0,
        num_rot_iter: int = 20
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Function to estimate the uncertainty (p-value) that a potential planet at
    a given position in a residual is not noise. The function supports
    several `tests <statistics.html>`_ and accounts for the effect of where the
    noise is extracted from the residual (see
    `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_)

    Args:
        frame: The residual on which we want to estimate the detection
            uncertainty.
        planet_position: The position of where the planet is expected. Can
            either be a position as (x_pos, y_pos) or
            (x_pos, y_pos, separation, angle).
            If photometry_mode_planet is set to a mode that supports search, the
            area around planet_position is explored to maximize the planet flux.
        statistical_test: The test used to constrain the planet flux
            needed to be counted as a detection.
            For the classical TTest (Gaussian noise) use an instance of
            :meth:`~applefy.statistics.parametric.TTest`. For Laplacian
            noise use
            :meth:`~applefy.statistics.bootstrapping.LaplaceBootstrapTest`.
        psf_fwhm_radius: The FWHM (radius) of the PSF. It is needed to
                sample independent noise values i.e. it determines the
                spacing between the noise observations which are extracted
                from the residuals.
        photometry_mode_planet: An instance of AperturePhotometryMode which
            defines how the flux is measured at the planet position.
        photometry_mode_noise: An instance of AperturePhotometryMode which
            defines how the noise photometry is measured.
        num_rot_iter: Number of tests performed with different positions of
            the noise values. See
            `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
            for more information.
        safety_margin: Area around the planet [pixel] which is excluded from
            the noise. This can be useful in case the planet has negative wings.

    Returns:
        1. The median of all num_rot_iter confidence level estimates
        (p-values /fpf).

        2. 1D numpy array with all num_rot_iter confidence level estimates
        (p-values /fpf).

        3. 1D numpy array with all num_rot_iter estimates of the test
        statistic T (SNR).
    """

    # 1.) make sure the photometry modes are compatible
    photometry_mode_planet.check_compatible(photometry_mode_noise)

    # 2.) estimate the planet flux
    planet_photometry = get_flux(frame=frame,
                                 position=planet_position,
                                 photometry_mode=photometry_mode_planet)[1]

    # 3.) Iterate over the noise
    noise_iterator = IterNoiseForPlanet(residual=frame,
                                        planet_position=planet_position[:2],
                                        safety_margin=safety_margin,
                                        psf_fwhm_radius=psf_fwhm_radius,
                                        num_rot_iter=num_rot_iter,
                                        max_rotation=None,
                                        photometry_mode=photometry_mode_noise)

    t_values = []
    p_values = []

    for tmp_noise_sample in noise_iterator:
        test_result = statistical_test.test_2samp(planet_photometry,
                                                  tmp_noise_sample)

        p_values.append(test_result[0])
        t_values.append(test_result[1])

    return float(np.median(p_values)), np.array(p_values), np.array(t_values)
