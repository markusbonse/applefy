import numpy as np
import pandas as pd
from scipy.ndimage import shift

from applefy.utils.positions import estimate_aperture_positions
from applefy.utils.general import center_subpixel


def calculate_fake_planet_positions(
        test_img,
        psf_fwhm_radius,
        num_planets=6,
        separations=None):
    """
    Function which estimates the positions for fake planets to be inserted to
    calculate contrast maps.

    Args:
        test_img: A 2D test image [np.array]
        psf_fwhm_radius: The radius of the PSF-FWHM [pixel].
        num_planets: The number of planets to be inserted (int). Has to be
            between 1 (minimum) and 6 (maximum). More planets provide more
            accuracy of the results.
        separations: Separations at which fake planets are inserted [pixel].
            By default, (If set to None) separations are selected in steps of
            1 lambda/D for the center to the edge of the test image:
                np.arange(0, center[0], aperture_radius * 2)[1:]

    Returns:
        planet_positions - dict which maps all separations (pixel) to a list of
            planet positions given as (x_pos, y_pos, separation, angle)

    """

    if num_planets > 7 or num_planets < 1:
        raise ValueError("The number of fake planets has to be between 1 and 6")

    center = center_subpixel(test_img)
    if separations is None:
        separations = np.arange(0, center[0], psf_fwhm_radius * 2)[1:]

    # results dicts
    planet_positions = dict()

    for tmp_separations in separations:
        tmp_positions = estimate_aperture_positions(
            tmp_separations,
            center=center,
            psf_fwhm_radius=psf_fwhm_radius)

        tmp_positions = np.array(tmp_positions)

        planet_idx = np.floor(
            np.linspace(0, tmp_positions.shape[0], num_planets, endpoint=False)
        ).astype(int)

        tmp_planet_positions = tmp_positions[planet_idx, :]

        planet_positions[tmp_separations] = \
            list(map(tuple, tmp_planet_positions))

    return planet_positions


def generate_fake_planet_experiments(
        flux_ratios: list,
        planet_positions: dict):
    """
        Function which creates config files for the contrast map. Each file
        corresponds to one experiment / data reduction. The experiment with idx
        0000 is the experiment with no fake planets.

    Args:
        flux_ratios:  A list of the planet-to-star flux_ratios used for the
            fake planets. If you want to calculate a simple contrast curve the
            list should contain a single value smaller than the expected
             detection limit. For the computation of a contrast grid several
             flux_ratios are needed.
        planet_positions: planet positions given by
            calculate_planet_apertures

    Returns: list of config files as dicts

    """

    all_config_files = dict()

    # --------------------------------------------------------------------------
    # create the reference experiment without fake planets
    exp_0_config = dict()
    exp_0_config["type"] = "FP estimation"
    exp_0_config["exp_id"] = "0000"
    all_config_files["0000"] = exp_0_config

    # --------------------------------------------------------------------------
    # Loop over all flux ratios and separations. Every tuple is one experiment.
    # These experiments are used to estimate the true positives as a
    # function of flux and separation

    exp_id_counter = 1

    for tmp_flux_ratio in flux_ratios:
        for tmp_separation in sorted(planet_positions.keys()):
            tmp_planets = planet_positions[tmp_separation]
            tmp_num_planets = len(tmp_planets)
            planet_names = ["a", "b", "c", "d", "e", "f"][:tmp_num_planets]

            for planet_idx, tmp_planet in enumerate(planet_names):
                tmp_exp_config = dict()
                tmp_exp_config["type"] = "TP estimation"

                tmp_exp_config["flux_ratio"] = tmp_flux_ratio
                tmp_exp_config["separation"] = tmp_separation

                tmp_exp_config["planet_position"] = tmp_planets[planet_idx]

                exp_id_name = str(exp_id_counter).zfill(4) + tmp_planet
                tmp_exp_config["exp_id"] = exp_id_name

                all_config_files[exp_id_name] = tmp_exp_config

            exp_id_counter += 1

    return all_config_files


def sort_fake_planet_results(results):
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


def add_fake_planets(
        input_stack: np.array,
        psf_template: np.array,
        parang: np.array,
        dit_science: float,
        dit_psf_template: float,
        experiment_config: dict,
        scaling_factor=1):
    """
    Function which adds fake planets to ADI data based on a contrast map
    config file.

    Args:
        input_stack: 3D numpy input stack (time, x, y) before normalization.
            Temporal-binning has to be done by mean not sum!
        psf_template: 2D numpy array of the unsaturated PSF used to create
            the fake planets. The template has to be in accordance to the
            integration time used for it. No normalization!
        parang: Parallactic angles 1D numpy array (rad)
        dit_science: integration time used for the science frames
        dit_psf_template: integration time used for the psf template
        experiment_config: configuration dict containing the information about
            where and how to add the fake planet
        scaling_factor: additional scaling factor e.g. needed for ND filter
            in M-band

    Returns: the input stack with the fake planet (3D numpy array)
    """

    # The IDX 0 case with no fake planets
    if "planet_position" not in experiment_config:
        return input_stack

    planet_position = experiment_config["planet_position"]

    # Pad or cut the template depending on the size of the science frames
    if psf_template.shape[-1] > input_stack.shape[-1]:
        cut_size = int((psf_template.shape[-1] -
                        input_stack.shape[-1]) / 2)

        padded_psf = psf_template[cut_size:-cut_size,
                                  cut_size:-cut_size]

    else:
        pad_size = int((input_stack.shape[-1] -
                        psf_template.shape[-1]) / 2)
        padded_psf = np.pad(psf_template,
                            pad_size, mode="constant",
                            constant_values=0)

    integration_time_factor = dit_science / dit_psf_template * scaling_factor

    # add the fake planet ------------------------------------------------------
    # Code inspired by PynPoint (Stolker et al 2019)

    # Scale the PSF to get a fake planet with the correct brightness
    flux_ratio = experiment_config["flux_ratio"]
    psf = padded_psf * integration_time_factor * flux_ratio

    # Calculate the positions of the fake planet a long time
    fake_planet_sep = planet_position[2]
    fake_planet_ang = np.radians(planet_position[3] - np.rad2deg(parang))
    x_shift = fake_planet_sep * np.cos(fake_planet_ang)
    y_shift = fake_planet_sep * np.sin(fake_planet_ang)

    # Shift the fake planet to the right position in the image
    im_shift = np.zeros(input_stack.shape)

    for i in range(input_stack.shape[0]):
        im_shift[i] = shift(
            psf,
            (float(y_shift[i]), float(x_shift[i])),
            order=5,
            mode="constant")

    return input_stack + im_shift


def merge_fake_planet_residuals(
        planet_dict,
        idx_table):
    """
    Function needed to merge residual frames collected with
    sort_results into one numpy array with dimensions:
        (num_separations, num_flux_ratios, num_planets, x, y)

    Args:
        planet_dict: A dictionary with keys = Experiment ID. For every ID a
            list is given which contains tuples of the residual and the position
            of the corresponding fake planet. See sort_results

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
