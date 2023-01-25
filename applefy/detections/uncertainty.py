import numpy as np

from applefy.statistics.general import TestInterface
from applefy.utils.photometry import AperturePhotometryMode, get_flux, \
    IterNoiseForPlanet


# TODO -> uncertainty
def compute_detection_confidence(
        frame,
        planet_position,
        test_statistic: TestInterface,
        psf_fwhm_radius,
        photometry_mode_planet: AperturePhotometryMode,
        photometry_mode_noise: AperturePhotometryMode,
        num_rot_iter=20,
        safety_margin=1.0):
    """
    Function to estimate the confidence that an observation at planet_position
    in a frame (residual or detection map) is actually a planet. The test
    supports all test statistics implemented in apelfei.statistics and accounts
    for the effect of where the reference positions are placed.

    Args:
        frame: The frame on which we want to estimate the confidence (2D array)
        planet_position: The position of where the planet is expected (pixel)
            If photometry_mode_planet is set to a mode that supports search the
            area around planet_position is explored to maximize the planet flux.
        test_statistic: The test statistic used
        psf_fwhm_radius: The size (radius) of resolution elements. It is used
            to sample independent noise values i.e. it sets the spacing between
            noise observations sampled from the fp_residual.
        photometry_mode_planet: An instance of AperturePhotometryMode which
            defines how the planet photometry is measured.
        photometry_mode_noise: An instance of AperturePhotometryMode which
            defines how the noise photometry is measured.
        num_rot_iter: Number of tests performed with different noise
            aperture positions. The classical Mawet et al. 2014 contrast curves
            do not consider this source of error.
        safety_margin: Area around the planet which is excluded from the noise.
            This can be useful in case the planet has negative wings.

    Returns:
        Tuple of:
            median_confidence_fpf - The median of all num_rot_iterations
                confidence level estimates (p-values /fpf)
            confidences_fpf - 1D numpy array with all num_rot_iterations
                confidence level estimates (p-values /fpf)
            tau_statistic - 1D numpy array with all num_rot_iterations
                estimates of the test statistic tau (SNR)
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
                                        num_iterations=num_rot_iter,
                                        max_rotation=None,
                                        photometry_mode=photometry_mode_noise)

    t_values = []
    p_values = []

    for tmp_noise_sample in noise_iterator:
        test_result = test_statistic.test_2samp(planet_photometry,
                                                tmp_noise_sample)

        p_values.append(test_result[0])
        t_values.append(test_result[1])

    return np.median(p_values), np.array(p_values), np.array(t_values)
