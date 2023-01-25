import multiprocessing
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from applefy.detections.uncertainty import compute_detection_uncertainty
from applefy.statistics.general import TestInterface, fpf_2_gaussian_sigma
from applefy.utils.photometry import AperturePhotometryMode, flux_ratio2mag, \
    mag2flux_ratio


def _compute_median_confidence(
        fake_planet_residuals,
        fake_planet_positions,
        separation,
        flux_ratio,
        test_statistic,
        psf_fwhm_radius,
        photometry_mode_planet,
        photometry_mode_noise,
        num_rot_iter=20,
        safety_margin=1.0):
    """
    Function used in compute_contrast_map to compute the  fpf of all fake
    planet residuals using multiprocessing. For more information about the
    input parameters check the documentation of the original function.
    """

    all_p_values = []
    for i in range(len(fake_planet_residuals)):
        tmp_fake_planet_residual = fake_planet_residuals[i]
        tmp_planet_position = fake_planet_positions[i]

        tmp_median_p, _, _ = compute_detection_uncertainty(
            frame=tmp_fake_planet_residual,
            planet_position=tmp_planet_position,
            statistical_test=test_statistic,
            psf_fwhm_radius=psf_fwhm_radius,
            photometry_mode_planet=photometry_mode_planet,
            photometry_mode_noise=photometry_mode_noise,
            num_rot_iter=num_rot_iter,
            safety_margin=safety_margin)

        all_p_values.append(tmp_median_p)

    print(".", end='')
    return separation, flux_ratio, np.median(all_p_values)


def compute_contrast_map(
        planet_dict,
        idx_table,
        test_statistic: TestInterface,
        psf_fwhm_radius,
        photometry_mode_planet: AperturePhotometryMode,
        photometry_mode_noise: AperturePhotometryMode,
        num_cores=1,
        num_rot_iter=20,
        safety_margin=1.0):
    """
    Computes a contrast map for a given set of fake planet residuals.
    Supports all statistical tests implemented in apelfei.statistics. Compared
    to the function compute_contrast_curve this function is applicable not only
    to residuals of PSF-subtraction methods (such as PCA), but in general. This
    allows to compare results of different methods from PSF-subtraction and
    Forward-modelling.

    Args:
        planet_dict: A dictionary with keys = Experiment ID. For every ID a
            list is given which contains tuples of the residual and the position
            of the corresponding fake planet. E.g.:

            planet_dict[1] = [(res_planet_a, pos_planet_a),
                              (res_planet_b, pos_planet_b),
                              (res_planet_c, pos_planet_c),
                              (res_planet_d, pos_planet_d),
                              (res_planet_e, pos_planet_e),
                              (res_planet_f, pos_planet_f)]
        idx_table: Pandas lookup table which links separation and flux_ratio
            to its experiment ID used by planet_dict
        test_statistic: The test statistic used to constrain the planet flux
            needed. For classical TTest curves use an instance of
            apelfei.statistics.parametric.TTest.
        psf_fwhm_radius: The size (radius) of resolution elements. It is used
            to sample independent noise values i.e. it sets the spacing between
            noise observations sampled from the fp_residual.
        photometry_mode_planet: An instance of AperturePhotometryMode which
            defines how the flux is measured at the planet positions.
        photometry_mode_noise: An instance of AperturePhotometryMode which
            defines how the noise photometry is measured.
        num_cores: Number of CPU cores used for a parallel computation of the
            map values.
        num_rot_iter: Number of tests performed with different noise
            aperture positions. The classical Mawet et al. 2014 contrast curves
            do not consider this source of error.
        safety_margin: Area around the planet which is excluded from the noise.
            This can be useful in case the planet has negative wings.

    Returns: The contrast map containing fpf values as a pandas DataFrame

    """

    # 1.) collect the data for multiprocessing
    all_parallel_experiments = []

    # Loop over idx map. Every tuple (separation_idx, flux_ratio_idx)
    # requires a separate calculation with multiprocessing
    for tmp_flux_ratio, tmp_separations in idx_table.items():
        for tmp_separation, exp_idx in tmp_separations.items():
            tmp_residuals = [i[0] for i in planet_dict[exp_idx]]
            tmp_planet_positions = [i[1] for i in planet_dict[exp_idx]]

            # save all parameters needed for multi-processing
            all_parallel_experiments.append((tmp_residuals,
                                             tmp_planet_positions,
                                             tmp_separation,
                                             tmp_flux_ratio,
                                             test_statistic,
                                             psf_fwhm_radius,
                                             photometry_mode_planet,
                                             photometry_mode_noise,
                                             num_rot_iter,
                                             safety_margin))

    # 2.) Run evaluations with multiprocessing
    pool = multiprocessing.Pool(processes=num_cores)
    print("Computing contrast map with multiprocessing:")
    mp_results = pool.starmap(_compute_median_confidence,
                              all_parallel_experiments)

    pool.close()
    pool.join()
    print("[DONE]")

    # 4.) Reshape and return the results
    results_combined = pd.DataFrame(np.array(mp_results),
                                    columns=["separation",
                                             "flux_ratio",
                                             "fpf"])

    contrast_map = results_combined.pivot_table(
        values="fpf",
        index="separation",
        columns="flux_ratio").sort_index(
        axis=1, ascending=False).T

    return contrast_map


def compute_contrast_from_map(
        contrast_map_fpf,
        fpf_threshold):
    """
    This function allows to obtain a contrast curve from a contrast map by
    interpolation and thresholding. Contains np.inf values in case the contrast
    can not be reached within the contrast map value range space.

    Args:
        contrast_map_fpf: The contrast map as 2D array as returned by
            compute_contrast_map. The flux ratios have to be in fraction not
            mag!
        fpf_threshold: The desired contrast as fpf

    Returns: A 1D pandas array containing the contrast curve

    """
    # Transform the fpf threshold to a threshold in terms of gaussian sigma.
    # This helps to get more accurate interpolation results
    threshold = fpf_2_gaussian_sigma(fpf_threshold)

    if contrast_map_fpf.index.values[0] > 1:
        raise ValueError("The contrast map flux ratios have to be in ratios "
                         " not magnitudes.")

    # we use interpolation in mag. This allows us to use in linspace
    local_contrast_map_fpf = deepcopy(contrast_map_fpf)
    local_contrast_map_fpf.index = flux_ratio2mag(local_contrast_map_fpf.index)

    contrast_mag = local_contrast_map_fpf.index.values
    interpolation_range = np.linspace(np.max(contrast_mag),
                                      np.min(contrast_mag),
                                      1000000)

    contrast_curve = []

    for tmp_sep, fpf_values in local_contrast_map_fpf.items():

        # Create the interpolation function
        tmp_contrast_func = interp1d(contrast_mag,
                                     fpf_2_gaussian_sigma(fpf_values),
                                     kind="linear")
        # interpolate the contrast map
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
