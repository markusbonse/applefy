"""
Functions needed to set up a performance map experiment
"""

import os
import json
import numpy as np

from applefy.utils.aperture_positions import \
    estimate_aperture_positions
from applefy.utils.general import center_subpixel


def calculate_planet_positions(
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
        flux_ratios:  List of flux_ratios to be studied [float,]
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


def save_experiment_configs(
        experimental_setups,
        experiment_config_dir,
        overwrite=False):

    # check if the config dir is empty
    if any([i.suffix == ".json" for i in experiment_config_dir.iterdir()]):
        if not overwrite:
            raise FileExistsError(
                "The directory \"" + str(experiment_config_dir) + "\" already "
                "contains config files. Delete them if you want to create a "
                "new experiment setup or use overwrite=True to automatically "
                " remove them.")

        print("Overwriting existing config files.")
        for tmp_file in experiment_config_dir.iterdir():
            if tmp_file.suffix == ".json":
                tmp_file.unlink()

    for tmp_id, tmp_config in experimental_setups.items():
        with open(os.path.join(
                experiment_config_dir,
                "exp_ID_" + str(tmp_id) + ".json"),
                'w') as f:
            json.dump(tmp_config,
                      f, indent=4)


def create_and_save_configs(test_img,
                            psf_fwhm_radius,
                            flux_ratios: list,
                            experiment_config_dir,
                            num_planets=6,
                            separations=None):
    """
    Function which creates several .json config files needed to run the
    experiments for the contrast map.

    Args:
        test_img: A 2D test image [np.array]
        psf_fwhm_radius: The radius of the PSF-FWHM [pixel].
        flux_ratios: List of flux_ratios to be studied [float,]
        experiment_config_dir: Destination where config files are stored
        num_planets: The number of planets to be inserted (int). Has to be
            between 1 (minimum) and 6 (maximum). More planets provide more
            accuracy of the results.
        separations: Separations at which fake planets are inserted [pixel].
            By default, (If set to None) separations are selected in steps of
            1 lambda/D for the center to the edge of the test image:
                np.arange(0, center[0], aperture_radius * 2)[1:]

    Returns: None

    """

    planet_positions = calculate_planet_positions(
        test_img,
        psf_fwhm_radius,
        num_planets=num_planets,
        separations=separations)

    experimental_setups = generate_fake_planet_experiments(
        flux_ratios,
        planet_positions)

    save_experiment_configs(
        experimental_setups=experimental_setups,
        experiment_root_dir=experiment_config_dir)
