"""
Function needed to evaluate results from fake planet experiments
"""
import os
import warnings
import json
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import multiprocessing

from applefy.utils.general import center_subpixel
from applefy.utils.data_handling import open_fits
from applefy.utils.aperture_photometry import get_flux, \
    IterNoiseBySeparation, IterNoiseForPlanet, AperturePhotometryMode
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.statistics.general import TestInterface, fpf_2_gaussian_sigma


################################################################################
# -------- Functions needed to read in data and evaluate the results -----------
################################################################################


def read_and_sort_results(result_files):
    """
    Function needed to read in the residuals and extract the meta-information
    from the .json config files.

    Args:
        result_files: List which contains the path to the residuals and
            corresponding
        config files. List items have to be structured like:
            (path to config file, path to the residual)

    Returns: A tuple containing elements (fp_residual, planet_dict, idx_table):
        - fp_residual: A 2D numpy array of the residual without any fake planets
        - planet_dict: A dictionary with keys = Experiment ID. For every ID a
            list is given which contains tuples of the residual and the position
            of the corresponding fake planet. E.g.:

            planet_dict[1] = [(res_planet_a, pos_planet_a),
                              (res_planet_b, pos_planet_b),
                              (res_planet_c, pos_planet_c),
                              (res_planet_d, pos_planet_d),
                              (res_planet_e, pos_planet_e),
                              (res_planet_f, pos_planet_f)]

        - idx_table: Pandas lookup table which links separation and flux_ratio
            to its experiment ID used by planet_dict
    """

    # 1. Load the results
    load_results = read_results(result_files)

    # 2. Sort the results
    return sort_results(load_results)


def read_results(result_files):
    load_results = []

    for tmp_input in result_files:
        tmp_config_path, tmp_residual_path = tmp_input

        # 1.) load the residual, ignore non-existing files
        if not os.path.isfile(tmp_residual_path):
            warnings.warn("File " + str(tmp_residual_path) + "not found")
            continue

        if not os.path.isfile(tmp_config_path):
            warnings.warn("File " + str(tmp_config_path) + "not found")
            continue

        tmp_residual = np.squeeze(open_fits(tmp_residual_path))

        # 1.1) load the config file
        with open(tmp_config_path) as json_file:
            tmp_setup_config = json.load(json_file)

        load_results.append(tmp_setup_config, tmp_residual)

    return load_results


def sort_results(results):
    # TODO change documentation
    """
    Function needed to read in the residuals and extract the meta-information
    from the .json config files.

    Args:
        results: List which contains the path to the residuals and corresponding
        config files. List items have to be structured like:
            (path to config file, path to the residual)

    Returns: A tuple containing elements (fp_residual, planet_dict, idx_table):
        - fp_residual: A 2D numpy array of the residual without any fake planets
        - planet_dict: A dictionary with keys = Experiment ID. For every ID a
            list is given which contains tuples of the residual and the position
            of the corresponding fake planet. E.g.:

            planet_dict[1] = [(res_planet_a, pos_planet_a),
                              (res_planet_b, pos_planet_b),
                              (res_planet_c, pos_planet_c),
                              (res_planet_d, pos_planet_d),
                              (res_planet_e, pos_planet_e),
                              (res_planet_f, pos_planet_f)]

        - idx_table: Pandas lookup table which links separation and flux_ratio
            to its experiment ID used by planet_dict
    """

    # FIRST STEP: Read in all config files and .fits residuals. The sorting step
    # has to be performed afterwards as the configs which have been computed are
    # unknown in advance

    fp_residual = None

    # collects all results based on their ID
    result_collection = dict()

    for tmp_input in results:
        # 1.) Unpack the current result
        tmp_setup_config, tmp_residual = tmp_input

        # 2.) Find out the experiment id
        tmp_exp_id = tmp_setup_config['exp_id']

        # 2.1) The 0000 id is the experiment that contains no planet
        if tmp_exp_id == "0000":
            fp_residual = tmp_residual
            continue

        # 2.2) All other experiments contain planets
        tmp_planet = tmp_exp_id[-1]
        tmp_exp_code = int(tmp_exp_id[:-1])

        # If we see the ID for the first time set up an empty list for the
        # residuals containing the fake planets
        if tmp_exp_code not in result_collection:
            result_collection[int(tmp_exp_code)] = []

        # store the result
        result_collection[int(tmp_exp_code)].append((tmp_planet,
                                                     tmp_setup_config,
                                                     tmp_residual))

    # SECOND STEP: Sort the collected results and bring them into a nice shape

    # We need this later to build the Index lookup table
    tmp_idx_to_sep_flux = dict()

    # dictionary to store the residuals and planet positions
    planet_dict = dict()

    for tmp_idx, tmp_tp_results in result_collection.items():
        # 1.) Extract the config data
        # Within each subgroup in result_collection the separation and
        # flux_ratio is the same. Use the first to extract it
        tmp_dataset_config = tmp_tp_results[0][1]  # config of first planet
        tmp_flux_ratio = tmp_dataset_config['flux_ratio']
        tmp_separation = tmp_dataset_config['separation']
        tmp_idx_to_sep_flux[tmp_idx] = (tmp_separation, tmp_flux_ratio)

        # 2.) Extract the residual data and planet position
        tmp_planets_and_pos = []
        for tmp_planet_result in tmp_tp_results:
            _, tmp_setup_config, tp_residual = tmp_planet_result
            tmp_planets_and_pos.append((tp_residual,
                                        tmp_setup_config["planet_position"]))
        planet_dict[tmp_idx] = tmp_planets_and_pos

    # 3.) Build idx table from config information
    idx_table = pd.DataFrame(tmp_idx_to_sep_flux,
                             ["separation", "flux_ratio"]).T.reset_index()
    idx_table = idx_table.pivot_table(values="index",
                                      index="separation",
                                      columns="flux_ratio")

    idx_table = idx_table.sort_index(axis=1, ascending=False).sort_index(axis=0)

    return fp_residual, planet_dict, idx_table


def estimate_stellar_flux(
        psf_template,
        dit_science,
        dit_psf_template,
        photometry_mode: AperturePhotometryMode,
        scaling_factor=1.0):
    """
    Function to estimate the normalized flux of the star given an unsaturated
    PSF image

    Args:
        psf_template: 2D array of the unsaturated PSF
        dit_science: Integration time of the science frames
        dit_psf_template: Integration time of unsaturated PSF
        scaling_factor: A scaling factor to account for ND filters.
        photometry_mode: An instance of AperturePhotometryMode which defines
            how the stellar flux is measured.

    Returns: The stellar flux (float)

    """

    # Account for the integration time of the psf template
    integration_time_factor = dit_science / dit_psf_template * scaling_factor
    psf_norm = psf_template * integration_time_factor

    center = center_subpixel(psf_norm)

    _, flux = get_flux(psf_norm, center,
                       photometry_mode=photometry_mode)

    return flux


def _compute_throughput_fixed_contrast(
        fake_planet_results,
        fp_residual,
        stellar_flux,
        inserted_flux_ratio,
        photometry_mode: AperturePhotometryMode):

    """
    Computes the throughput for a given list of fake planet residuals and
    flux ratio. The brightness of the inserted fake planets has to be fixed.
    In general the function compute_throughput_table should be used instead of
    this function.

    Args:
        fake_planet_results: List containing the residuals (2d np.array) and
            position (tuple of floats) of the fake planet experiments. Format:
            [(res_planet_a, pos_planet_a), (res_planet_b, pos_planet_b),...]
        fp_residual: A 2D numpy array of the residual without any fake planets
        stellar_flux: The stellar flux estimated by estimate_stellar_flux.
            Note: The settings for the used computation mode have to be the same
            as used by this function
        inserted_flux_ratio: The flux ratio of the fake planets inserted (float)
            Has to be the same for all input residuals
        photometry_mode: An instance of AperturePhotometryMode which defines
            how the flux is measured at the planet positions.

    Returns: List of throughput values for every planet
    """

    flux_in = stellar_flux * inserted_flux_ratio
    all_throughput_results = []

    # Loop over all residuals
    for tmp_planet_result in fake_planet_results:
        tmp_planet_residual, tmp_position = tmp_planet_result

        # Note: This line assumes that a linear PSF-subtraction method was used
        planet_signal_offset = tmp_planet_residual - fp_residual

        # sometimes negative self-subtraction wings are very close to the
        # planet. In oder to remove this effect we set these values to 0
        planet_signal_offset[planet_signal_offset < 0] = 0

        _, tmp_flux_out = get_flux(planet_signal_offset,
                                   tmp_position[:2],
                                   photometry_mode=photometry_mode)

        tmp_throughput = tmp_flux_out / flux_in

        # negative throughput can happen due to numerical problems
        if tmp_throughput < 0:
            tmp_throughput = 0

        all_throughput_results.append(tmp_throughput)

    return all_throughput_results


def compute_throughput_table(
        planet_dict,
        fp_residual,
        idx_table,
        stellar_flux,
        photometry_mode_planet: AperturePhotometryMode):
    """
    Computes the throughput for a series of experiments as generated by function
    read_and_sort_results.

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
        fp_residual: A 2D numpy array of the residual without any fake planets
        idx_table: Pandas lookup table which links separation and flux_ratio
            to its experiment ID used by planet_dict
        stellar_flux: The stellar flux estimated by estimate_stellar_flux.
            Note: The settings for the used computation mode have to be the same
            as used by this function
        photometry_mode_planet: An instance of AperturePhotometryMode which
            defines how the flux is measured at the planet positions.

    Returns: The throughput for all fake planet experiments as dict and table:
        -  throughput_dict: A dictionary with keys = Experiment ID. For every ID
            a list is given which contains the throughput for every residual
        - throughput_table: Pandas table which contains the median throughput
            as a function of separation and inserted flux_ratio
    """

    throughput_dict = dict()
    throughput_table = idx_table.copy()

    for separation, row in idx_table.iterrows():
        for tmp_flux_ratio, tmp_idx in row.items():

            tmp_fake_planet_results = planet_dict[tmp_idx]

            tmp_throughput_list = _compute_throughput_fixed_contrast(
                tmp_fake_planet_results,
                fp_residual,
                stellar_flux,
                tmp_flux_ratio,
                photometry_mode=photometry_mode_planet)

            throughput_dict[tmp_idx] = tmp_throughput_list
            throughput_table.loc[separation, tmp_flux_ratio] =\
                np.median(tmp_throughput_list)

    return throughput_dict, throughput_table


def merge_residual_stack(
        planet_dict,
        idx_table):
    """
    Function needed to merge residual frames collected with
    read_and_sort_results into one numpy array with dimensions:
        (num_separations, num_flux_ratios, num_planets, x, y)

    Args:
        planet_dict: A dictionary with keys = Experiment ID. For every ID a
            list is given which contains tuples of the residual and the position
            of the corresponding fake planet. See read_and_sort_results

        idx_table: Pandas lookup table which links separation and flux_ratio
            to its experiment ID used by planet_dict

    Returns:

    """
    residuals_paired = np.array([[y[0] for y in x]
                                 for _, x in planet_dict.items()])

    all_residuals_sorted = []
    for separation, row in idx_table.iterrows():
        tmp_row_residuals = []
        for tmp_idx in row:
            tmp_row_residuals.append(residuals_paired[tmp_idx - 1])

        all_residuals_sorted.append(tmp_row_residuals)

    return np.array(all_residuals_sorted)


################################################################################
# -------------------- Functions to calculate contrast -------------------------
################################################################################

def compute_contrast_curve(
        throughput_list: pd.DataFrame,
        stellar_flux,
        fp_residual,
        confidence_level_fpf,
        test_statistic: TestInterface,
        psf_fwhm_radius,
        photometry_mode_noise,
        num_rot_iter=100):

    """
    Computes a contrast curve by using throughput values and solving for the
    planet flux needed to reach a given confidence. Supports all statistical
    tests implemented in apelfei.statistics.

    Args:
        throughput_list: 1D pandas array of throughput values for every
            separation. (floats in range [0, 1]).
        stellar_flux: The brightness of the star. The mode used to estimate the
            stellar_flux has to be compatible with the photometry_mode_noise
            used here.
        fp_residual: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the noise statistics. This is usually the
            residual without fake planets.
        confidence_level_fpf: The confidence level associated with the contrast
            curve. Can be a single fpf value (float) or a list of fpf values for
            every separation. In case a list is given the number of fpf values
            has to match the number of values in the throughput_list.
        test_statistic: The test statistic used to constrain the planet flux
            needed. For classical TTest curves use an instance of
            apelfei.statistics.parametric.TTest.
        psf_fwhm_radius: The size (radius) of resolution elements. It is used
            to sample independent noise values i.e. it sets the spacing between
            noise observations sampled from the fp_residual.
        photometry_mode_noise: An instance of AperturePhotometryMode which
            defines how the noise photometry is measured.
        num_rot_iter: Number of tests performed with different noise
            aperture positions. The classical Mawet et al. 2014 contrast curves
            do not consider this source of error.

    Returns:
        Tuple of:
            median_contrast_curve - The median of all num_rot_iterations
                contrast curves
            mad_contrast_curve_error - The median absolute deviation of all
                num_rot_iterations contrast curves
            contrast_curves - A pandas DataFrame containing all
                num_rot_iterations contrast curves calculated.
    """

    # 1.) check confidence_level
    if isinstance(confidence_level_fpf, list):
        assert len(confidence_level_fpf) == throughput_list.shape[0]
    elif not isinstance(confidence_level_fpf, float):
        raise ValueError("confidence_level has to be a list or float of fpf")

    # 2.) compute the contrast curves
    # We compute one contrast curve for every rotation of noise positions
    contrast_values = []

    # Loop over throughput_list.
    for sep_idx, (tmp_separation, tmp_throughput) in tqdm(
            enumerate(throughput_list.items()),
            total=len(throughput_list)):

        if isinstance(confidence_level_fpf, list):
            tmp_fpf = confidence_level_fpf[sep_idx]
        else:
            tmp_fpf = confidence_level_fpf

        # iterate over the fp residual to get the noise
        noise_iterator = IterNoiseBySeparation(
            residual=fp_residual,
            separation=tmp_separation,
            psf_fwhm_radius=psf_fwhm_radius,
            num_iterations=num_rot_iter,
            photometry_mode=photometry_mode_noise,
            max_rotation=360)

        # Run evaluate contrast for num_rot_iter positions
        for rot_idx, tmp_observations in enumerate(noise_iterator):
            tmp_noise_at_planet = tmp_observations[0]
            tmp_noise_sample = tmp_observations[1:]

            residual_flux_needed = test_statistic.constrain_planet(
                tmp_noise_at_planet,
                tmp_noise_sample,
                tmp_fpf)

            # it is possible that the tmp_throughput is almost 0 which can lead
            # to invalid division by zero (numerical stability)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flux_throughput_corr = residual_flux_needed / tmp_throughput

            tmp_contrast = flux_throughput_corr / stellar_flux

            contrast_values.append((tmp_separation, tmp_contrast, rot_idx))

    # 3.) Construct the result tables
    contrast_curves = pd.DataFrame(contrast_values,
                                   columns=["separation", "contrast",
                                            "rotation_idx"])

    contrast_curves = contrast_curves.pivot(
        index="rotation_idx",
        columns="separation")

    median_contrast = np.median(contrast_curves, axis=0)
    # it is possible that the contrast_curves contains inf values due to very
    # small throughput values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contrast_mad = stats.median_abs_deviation(contrast_curves, axis=0)

    return median_contrast, contrast_mad, contrast_curves


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

        tmp_median_p, _, _ = compute_detection_confidence(
            frame=tmp_fake_planet_residual,
            planet_position=tmp_planet_position,
            test_statistic=test_statistic,
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


################################################################################
# ----------------------- Evaluation wrapped into classes ----------------------
################################################################################

class ContrastResult(object):
    """
    Wrapper class for the evaluation and organization of residuals from one
    method (e.g. pca with 10 components). Supports both contrast curves and
    contrast maps.
    """

    def __init__(
            self,
            model_results,
            stellar_flux,
            planet_photometry_mode: AperturePhotometryMode,
            noise_photometry_mode: AperturePhotometryMode,
            psf_fwhm_radius):
        # TODO change documentation
        """
        Constructor of the class. This function will read in all residuals and
        compute the throughput table.

        Args:
            model_results: List which contains the path to the residuals and
                corresponding config files. List items have to be structured
                like: (path to config file, path to the residual)
            stellar_flux: The stellar flux measured with estimate_stellar_flux.
                The mode used to get the stellar flux has to be the same as used
                given in planet_mode.
            planet_photometry_mode: An instance of AperturePhotometryMode which
                defines how the flux is measured at the planet positions.
            noise_photometry_mode: An instance of AperturePhotometryMode which
                defines how the flux is measured at the noise positions.
            psf_fwhm_radius: Determines the spacing between residual elements.
                Usually, it is the radius of one FWHM (pixel).
        """

        # Init additional members needed for flux based estimations
        self.stellar_flux = stellar_flux

        # Check if photometry_modes are compatible
        if not planet_photometry_mode.check_compatible(noise_photometry_mode):
            raise ValueError("Photometry modes " +
                             planet_photometry_mode.m_flux_mode + " and " +
                             noise_photometry_mode.m_flux_mode + " are not"
                             " compatible.")

        # Save the inputs
        self.psf_fwhm_radius = psf_fwhm_radius
        self.planet_mode = planet_photometry_mode
        self.noise_mode = noise_photometry_mode

        # Read in the results
        read_in = sort_results(model_results)
        self.fp_residual, self.planet_dict, self.idx_table = read_in

        # In case throughput values are computed later we initialize the member
        # variables here
        self.throughput_dict = None
        self.median_throughput_table = None

    def compute_throughput(self):
        """
        Computes the throughput table and saves it internally.

        Returns: Pandas table which contains the median throughput
            as a function of separation and inserted flux_ratio
        """

        if self.median_throughput_table is not None:
            return self.median_throughput_table

        self.throughput_dict, self.median_throughput_table = \
            compute_throughput_table(self.planet_dict,
                                     self.fp_residual,
                                     self.idx_table,
                                     self.stellar_flux,
                                     photometry_mode_planet=self.planet_mode)

        return self.median_throughput_table

    @property
    def residuals(self):
        """
        Returns: A numpy array which combines all residuals from all experiments
            Shape (num_separations, num_flux_ratios, num_planets, x, y)
        """
        return merge_residual_stack(self.planet_dict,
                                    self.idx_table)

    def compute_contrast_curve(
            self,
            confidence_level_fpf,
            test_statistic,
            num_rot_iter=100):
        """
        Computes a contrast curve given a confidence levels and test statistic.
        This is the analytic computation of a contrast curve that is only
        consistent with PSF-subtraction algorithms such as PCA. For other
        algorithms compute a contrast_map.

        Args:
            confidence_level_fpf: A single fpf value (float) or a list of fpf
                values. In case a list is given the number of fpf values has to
                match the number of evaluated separations.
            test_statistic: The test statistic used to constrain the planet flux
                needed. For classical TTest curves use an instance of
                apelfei.statistics.parametric.TTest.
            num_rot_iter: Number of tests performed with different noise
                positions.

        Returns: Tuple of:
            - A 1D table containing the median contrast values.
            - A 1D table containing the mad-error of the contrast curves caused
                by the positioning of the noise element positions.
        """

        if self.median_throughput_table is None:
            self.compute_throughput()

        median_contrast_curve, contrast_error, _ = compute_contrast_curve(
            # use the last row of the throughput table as the throughput
            throughput_list=self.median_throughput_table.T.iloc[-1],
            stellar_flux=self.stellar_flux,
            fp_residual=self.fp_residual,
            confidence_level_fpf=confidence_level_fpf,
            test_statistic=test_statistic,
            psf_fwhm_radius=self.psf_fwhm_radius,
            photometry_mode_noise=self.noise_mode,
            num_rot_iter=num_rot_iter)

        # wrap contrast curves into pandas arrays
        median_contrast_curve = pd.DataFrame(
            median_contrast_curve,
            index=self.idx_table.index,
            columns=["contrast", ])

        contrast_error = pd.DataFrame(
            contrast_error,
            index=self.idx_table.index,
            columns=["MAD of contrast", ])

        return median_contrast_curve, contrast_error

    def compute_contrast_grid(
            self,
            test_statistic,
            num_cores=1,
            num_rot_iter=20,
            safety_margin=1.0,
            confidence_level_fpf=None):
        """
        Function which calculates the contrast map i.e. how confident are we,
        that a certain detection is possible at separations s and flux_ratio f.
        The planet flux is given by stellar_flux * flux_ratio * throughput.
        The test accounts for effects caused by the rotation of reference
        aperture positions during the SNR estimation.

        Args:
            test_statistic: The test statistic used to constrain the planet flux
                needed. For classical TTest curves use an instance of
                apelfei.statistics.parametric.TTest.
            num_cores: Number of parallel processes used to calculate the
                contrast map.
            num_rot_iter: Number of tests performed with different noise
                positions.
            safety_margin: Area around the planet which is excluded from the
                noise. This can be useful in case the planet has negative wings.
            confidence_level_fpf: If set to a float value the output contrast
                map will be interpolated and transformed into a contrast curve.
                if None only the contrast map is returned.

        Returns: The contrast map for the chosen test. We report the median
            p-values over all num_rot_iterations experiments performed.
            If contrast_curve_fpf is a float a 1D pandas array containing the
            contrast curve is returned as well.

        """

        contrast_map = compute_contrast_map(
            planet_dict=self.planet_dict,
            idx_table=self.idx_table,
            test_statistic=test_statistic,
            psf_fwhm_radius=self.psf_fwhm_radius,
            photometry_mode_planet=self.planet_mode,
            photometry_mode_noise=self.noise_mode,
            num_cores=num_cores,
            num_rot_iter=num_rot_iter,
            safety_margin=safety_margin)

        if isinstance(confidence_level_fpf, (float, np.floating)):
            # compute the contrast curve
            contrast_curve = compute_contrast_from_map(
                contrast_map,
                confidence_level_fpf)

            return contrast_map, contrast_curve

        return contrast_map
