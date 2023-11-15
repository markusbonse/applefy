"""
Util functions needed to compute contrast grids. The recommended way
to compute contrast grids is to use the class
:meth:`~applefy.detections.contrast.Contrast` and not the util functions.
"""

from typing import Tuple, List, Dict, Union

import multiprocessing
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from applefy.detections.uncertainty import compute_detection_uncertainty
from applefy.statistics.general import TestInterface, fpf_2_gaussian_sigma
from applefy.utils.photometry import AperturePhotometryMode, flux_ratio2mag, \
    mag2flux_ratio, IterNoiseBySeparation, get_flux


def _compute_detection_uncertainty_throughput(
        fake_planet_residual: np.ndarray,
        fp_residual: np.ndarray,
        planet_position: Union[Tuple[float, float],
                                     Tuple[float, float, float, float]],
        separation: float,
        statistical_test: TestInterface,
        psf_fwhm_radius: float,
        photometry_mode_planet: AperturePhotometryMode,
        photometry_mode_noise: AperturePhotometryMode,
        num_rot_iter: int = 20):
    """
    Function used in compute_contrast_grid to compute the fpf of all fake
    planet residuals using multiprocessing. For more information about the
    input parameters check the documentation of the main function.
    """

    # 1.) Estimate the flux of the planet without noise
    # Note: This line assumes that a linear PSF-subtraction method was used
    planet_signal_offset = fake_planet_residual - fp_residual

    # sometimes negative self-subtraction wings are very close to the
    # planet. In oder to remove this effect we set these values to 0
    planet_signal_offset[planet_signal_offset < 0] = 0

    # get the flux of the planet
    _, planet_flux = get_flux(
        planet_signal_offset,
        planet_position[:2],
        photometry_mode=photometry_mode_planet)

    # 2.) Iterate over different noise positions
    # iterate over the fp residual to get the noise
    noise_iterator = IterNoiseBySeparation(
        residual=fp_residual,
        separation=separation,
        psf_fwhm_radius=psf_fwhm_radius,
        num_rot_iter=num_rot_iter,
        photometry_mode=photometry_mode_noise,
        max_rotation=360)

    t_values = []
    p_values = []
    for rot_idx, tmp_observations in enumerate(noise_iterator):
        tmp_noise_at_planet = tmp_observations[0]
        tmp_noise_sample = tmp_observations[1:]

        # add the planet flux to the noise at the position of the planet
        tmp_planet_flux = tmp_noise_at_planet + planet_flux

        # run the statistical test
        test_result = statistical_test.test_2samp(
            tmp_planet_flux, tmp_noise_sample)

        p_values.append(test_result[0])
        t_values.append(test_result[1])

    return float(np.median(p_values)), np.array(p_values), np.array(t_values)


def _compute_median_confidence(
        fake_planet_residuals: np.ndarray,
        fp_residual: np.ndarray,
        fake_planet_positions: List[Tuple[float, float]],
        separation: float,
        flux_ratio: float,
        statistical_test: TestInterface,
        psf_fwhm_radius: float,
        photometry_mode_planet: AperturePhotometryMode,
        photometry_mode_noise: AperturePhotometryMode,
        compute_snr_instead: bool = False,
        num_rot_iter: int = 20,
        safety_margin: float = 1.0
) -> Tuple[float, float, float]:
    """
    Function used in compute_contrast_grid to compute the fpf of all fake
    planet residuals using multiprocessing. For more information about the
    input parameters check the documentation of the main function.
    """

    all_p_values = []
    for i, tmp_fake_planet_residual in enumerate(fake_planet_residuals):
        tmp_planet_position = fake_planet_positions[i]

        if safety_margin == -1:
            tmp_median_p, _, tmp_t_values = _compute_detection_uncertainty_throughput(
                fake_planet_residual=tmp_fake_planet_residual,
                fp_residual=fp_residual,
                planet_position=tmp_planet_position,
                separation=separation,
                statistical_test=statistical_test,
                psf_fwhm_radius=psf_fwhm_radius,
                photometry_mode_planet=photometry_mode_planet,
                photometry_mode_noise=photometry_mode_noise,
                num_rot_iter=num_rot_iter)
        else:
            tmp_median_p, _, tmp_t_values = compute_detection_uncertainty(
                frame=tmp_fake_planet_residual,
                planet_position=tmp_planet_position,
                statistical_test=statistical_test,
                psf_fwhm_radius=psf_fwhm_radius,
                photometry_mode_planet=photometry_mode_planet,
                photometry_mode_noise=photometry_mode_noise,
                num_rot_iter=num_rot_iter,
                safety_margin=safety_margin)

        if compute_snr_instead:
            all_p_values.append(np.median(tmp_t_values))
        else:
            all_p_values.append(tmp_median_p)

    print(".", end='')
    return separation, flux_ratio, float(np.median(all_p_values))


def compute_contrast_grid(
        planet_dict: Dict[int,
                          List[Tuple[np.ndarray,
                                     List[float]]]],
        idx_table: pd.DataFrame,
        fp_residual: np.ndarray,
        statistical_test: TestInterface,
        psf_fwhm_radius: float,
        photometry_mode_planet: AperturePhotometryMode,
        photometry_mode_noise: AperturePhotometryMode,
        compute_snr_grid: bool = False,
        num_cores: int = 1,
        num_rot_iter: int = 20,
        safety_margin: float = 1.0
) -> pd.DataFrame:
    """
    Computes a contrast grid for a given set of fake planet residuals.
    Supports all statistical tests implemented in
    `Statistics <statistics.html>`_. Compared to the function
    :meth:`~applefy.utils.contrast_curve.compute_contrast_curve` this function
    is applicable not only to residuals of linear PSF-subtraction methods
    like PCA, but in general. This allows to compare results of different
    methods.

    Args:
        planet_dict: A dictionary with keys = Experiment ID. For every ID a
            list is given which contains tuples of the residual and the position
            of the corresponding fake planet. E.g.:

            .. highlight:: python
            .. code-block:: python

                planet_dict["0001"] = [
                    (res_planet_a, pos_planet_a),
                    (res_planet_b, pos_planet_b),
                    (res_planet_c, pos_planet_c),
                    (res_planet_d, pos_planet_d),
                    (res_planet_e, pos_planet_e),
                    (res_planet_f, pos_planet_f)]

        idx_table: Pandas table which links separation and flux_ratio
            to its experiment ID as used by planet_dict.
        fp_residual: Residual without the fake planet. This is used if
            safety_margin is set to -1. In this case the noise is extracted
            from the fp_residual instead of the fake planet residuals.
        statistical_test: The test used to constrain the planet flux
            needed to be counted as a detection.
            For the classical TTest (Gaussian noise) use an instance of
            :meth:`~applefy.statistics.parametric.TTest`. For Laplacian
            noise use
            :meth:`~applefy.statistics.bootstrapping.LaplaceBootstrapTest`.
        psf_fwhm_radius: The FWHM (radius) of the PSF. It is needed to
            sample independent noise values i.e. it determines the
            spacing between the noise observations which are extracted
            from the fp_residual.
        photometry_mode_planet: An instance of AperturePhotometryMode which
            defines how the flux is measured at the planet positions.
        photometry_mode_noise: An instance of AperturePhotometryMode which
            defines how the noise photometry is measured.
        compute_snr_grid: If set to True the values of the grid will contain
            the SNR i.e. value of the test statistic instead of the fpf.
        num_cores: Number of CPU cores used for a parallel computation of the
            grid values.
        num_rot_iter: Number of tests performed with different positions of the
            noise values. See `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_ for more information.
        safety_margin: Area around the planet [pixel] which is excluded from
            the noise. This can be useful in case the planet has negative wings.
            Can be set to -1. In this case the noise is extracted from the
            residual without the fake planet.

    Returns:
        The contrast grid containing p-values / fpf values as a pandas DataFrame.
        If compute_snr_grid is set to True the values of the grid will contain
        the SNR i.e. value of the test statistic instead of the fpf.

    """
    if compute_snr_grid:
        output_tag = "SNR"
    else:
        output_tag = "fpf"

    # 1.) collect the data for multiprocessing
    all_parallel_experiments = []

    # Loop over idx grid. Every tuple (separation_idx, flux_ratio_idx)
    # requires a separate calculation with multiprocessing
    for tmp_flux_ratio, tmp_separations in idx_table.items():
        for tmp_separation, exp_idx in tmp_separations.items():
            tmp_residuals = [i[0] for i in planet_dict[exp_idx]]
            tmp_planet_positions = [i[1] for i in planet_dict[exp_idx]]

            # save all parameters needed for multi-processing
            all_parallel_experiments.append((tmp_residuals,
                                             fp_residual,
                                             tmp_planet_positions,
                                             tmp_separation,
                                             tmp_flux_ratio,
                                             statistical_test,
                                             psf_fwhm_radius,
                                             photometry_mode_planet,
                                             photometry_mode_noise,
                                             compute_snr_grid,
                                             num_rot_iter,
                                             safety_margin))

    # 2.) Run evaluations with multiprocessing
    with multiprocessing.Pool(processes=num_cores) as pool:
        print("Computing contrast grid with multiprocessing:")
        mp_results = pool.starmap(_compute_median_confidence,
                                  all_parallel_experiments)
    print("[DONE]")

    # 4.) Reshape and return the results
    results_combined = pd.DataFrame(np.array(mp_results),
                                    columns=["separation",
                                             "flux_ratio",
                                             output_tag])

    contrast_grid = results_combined.pivot_table(
        values=output_tag,
        index="separation",
        columns="flux_ratio").sort_index(
        axis=1, ascending=False).T

    return contrast_grid


def compute_contrast_from_grid(
        contrast_grid_fpf: pd.DataFrame,
        fpf_threshold: float
) -> pd.DataFrame:
    """
    This function allows to obtain a contrast curve from a contrast grid by
    interpolation and thresholding. Contains np.inf values in case the contrast
    can not be reached within the contrast grid.

    Args:
        contrast_grid_fpf: The contrast grid as 2D pandas Table as returned by
            :meth:`~compute_contrast_grid`. The flux ratios have to be in
            fraction not mag!
        fpf_threshold: The desired detection threshold as fpf.

    Returns:
        A 1D pandas array containing the contrast curve.

    """
    # Transform the fpf threshold to a threshold in terms of gaussian sigma.
    # This helps to get more accurate interpolation results
    threshold = fpf_2_gaussian_sigma(fpf_threshold)

    if contrast_grid_fpf.index.values[0] > 1:
        raise ValueError("The contrast grid flux ratios have to be in ratios "
                         " not magnitudes.")

    # we use interpolation in mag. This allows us to use in linspace
    local_contrast_grid_fpf = deepcopy(contrast_grid_fpf)
    local_contrast_grid_fpf.index = flux_ratio2mag(local_contrast_grid_fpf.index)

    contrast_mag = local_contrast_grid_fpf.index.values
    interpolation_range = np.linspace(np.max(contrast_mag),
                                      np.min(contrast_mag),
                                      1000000)

    contrast_curve = []

    for tmp_sep, fpf_values in local_contrast_grid_fpf.items():

        # Create the interpolation function
        tmp_contrast_func = interp1d(
            contrast_mag,
            fpf_2_gaussian_sigma(fpf_values.values),
            kind="linear")

        # interpolate the contrast grid
        tmp_interpolated_results = tmp_contrast_func(interpolation_range)

        # find the first value above the threshold
        tmp_contrast_threshold_idx = np.argmax(
            tmp_interpolated_results > threshold)

        # If no values exists that exceeds the threshold return -inf
        if tmp_contrast_threshold_idx == 0:
            contrast_curve.append((-np.inf,
                                   tmp_sep))
        else:
            contrast_curve.append(
                (mag2flux_ratio(interpolation_range[
                                    tmp_contrast_threshold_idx]),
                 tmp_sep))

    # Creates a pandas table for the results and return it
    # convert the results back to flux ratios
    contrast_curve_pd = pd.DataFrame(
        np.array(contrast_curve),
        columns=["contrast", "separation"])

    return contrast_curve_pd.set_index("separation")
