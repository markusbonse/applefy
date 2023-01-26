"""
Simple helper functions to handle files.
"""

import os
import json
import warnings
from typing import List, Dict, Tuple, Union
from copy import deepcopy

from pathlib import Path
import h5py
from astropy.io import fits
import numpy as np


def read_apples_with_apples_root() -> str:
    """
    A simple function which reads in the APPLES_ROOT_DIR specified by the user.
    This function is needed to reproduce the results of the Apples with Apples
    paper. Raises an error if the directory does not exist.

    Returns:
        The path to the root directory
    """
    root_dir = os.getenv('APPLES_ROOT_DIR')

    if not os.path.isdir(root_dir):
        raise IOError("The path in APPLES_ROOT_DIR does not exist. Make sure "
                      "to download the data an specify its location.")

    print("Data in the APPLES_ROOT_DIR found. Location: " + str(root_dir))
    return root_dir


def create_checkpoint_folders(
        checkpoint_dir: Union[str, Path]
) -> Union[None, Tuple[Path, Path, Path]]:
    """
    This function create the classical checkpoint folder structure as used by
    :meth:`~applefy.detections.contrast.Contrast`. It creates three
    sub-folders. configs_cgrid, residuals, scratch. Returns None if
    checkpoint_dir is None.

    Args:
        checkpoint_dir: The root directory in which the sub-folders are created.

    Returns:
        1. Path to the configs_cgrid folder.
        2. Path to the residuals folder.
        3. Path to the scratch folder.

    """

    # if no experiment_root_dir is given we don't save results
    if checkpoint_dir is None:
        return None

    # use pathlib for easy path handling
    checkpoint_dir = Path(checkpoint_dir)

    # check if the experiment_root_dir exists
    if not checkpoint_dir.is_dir():
        raise IOError("The directory " + str(checkpoint_dir) +
                      " does not exist. Please create it.")

    # create sub-folders if they do not exist
    config_dir = checkpoint_dir / "configs_cgrid"
    residual_dir = checkpoint_dir / "residuals"
    scratch_dir = checkpoint_dir / "scratch"

    config_dir.mkdir(parents=False, exist_ok=True)
    residual_dir.mkdir(parents=False, exist_ok=True)
    scratch_dir.mkdir(parents=False, exist_ok=True)

    return config_dir, residual_dir, scratch_dir


def search_for_config_and_residual_files(
        config_dir: Union[str, Path],
        method_dir: Path
) -> List[Tuple[str, str]]:
    """
    Searches for tuples of existing contrast grid config files and
    corresponding residuals. Raises an Error if the .json files and residual
    files in config_dir and method_dir do not match.

    Args:
        config_dir: Directory where the contrast grid config files are stored.
        method_dir: Directory where the residuals are stored.

    Returns:
        A list with paired paths (config path, residual path)
    """

    collected_result_file = []

    # find all config files
    config_files = dict(collect_all_contrast_grid_configs(config_dir))

    for tmp_file in method_dir.iterdir():
        if not tmp_file.name.startswith("residual_"):
            continue

        tmp_idx = tmp_file.name.split("_ID_")[1].split(".")[0]
        tmp_config_path = str(config_files[tmp_idx])
        tmp_residual_path = str(tmp_file)

        del config_files[tmp_idx]
        collected_result_file.append((tmp_config_path, tmp_residual_path))

    if len(config_files) != 0:
        raise FileNotFoundError(
            "Some residuals are missing. Check if all config files "
            "have a matching residual.")

    return collected_result_file


def collect_all_contrast_grid_configs(
        contrast_grid_configs: str
) -> List[Tuple[str, str]]:
    """
    Simple function which looks for all auto generated contrast grid config
    files in one directory.

    Args:
        contrast_grid_configs: The directory which contains the config files.

    Returns:
        A list of tuples (job_id, file path)
    """

    # 1.) Collect all jobs to be run
    all_datasets_configs = []
    for tmp_file in sorted(os.listdir(contrast_grid_configs)):
        if not tmp_file.startswith("exp_"):
            continue

        tmp_id = tmp_file.split(".")[0].split("_")[-1]
        all_datasets_configs.append(
            (tmp_id, os.path.join(contrast_grid_configs, tmp_file)))

    return all_datasets_configs


def save_experiment_configs(
        experimental_setups: Dict[str, Dict],
        experiment_config_dir: Path,
        overwrite: bool = False
) -> None:
    """
    Saves all contrast grid config files in experimental_setups as .json files
    into the experiment_config_dir. Overwrites existing files is requested.

    Args:
        experimental_setups: The contrast grid config files as created by
            :meth:`~applefy.detections.preparation.generate_fake_planet_experiments`
        experiment_config_dir: The directory where the .json files are saved.
        overwrite: Whether to overwrite existing config files.
    """

    # check if the config dir is empty
    if any(i.suffix == ".json" for i in experiment_config_dir.iterdir()):
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
                'w') as file:
            json.dump(tmp_config, file, indent=4)


def read_fake_planet_results(
        result_files: List[Tuple[str, str]]
) -> List[Tuple[dict, np.ndarray]]:
    """
    Read all contrast grid config files and .fits residuals as listed in
    result_files.

    Args:
        result_files: A list of tuples
             (path to config file, path to residual)

    Returns:
        The load results as a list of
             (config, residual)

    """

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

        load_results.append((tmp_setup_config, tmp_residual))

    return load_results


def cut_around_center(
        psf_template: np.array,
        science_data: np.array,
        cut_off_psf: int,
        cut_off_data: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function which cuts the psf_template and science_data around the center of
    the image. Can be useful to reduce the dimensionality of the data in order
    to save time during the calculation of contrast grids and contrast curves.

    Args:
        psf_template: 2D numpy array with the pst template.
        science_data: 2D numpy array with the science data. Shape (time, x, y)
        cut_off_psf: The number of pixel to be cut off on each side of the
            pst template.
        cut_off_data: The number of pixel to be cut off on each side of the
            science data.

    Returns:
        1. The cut science data.
        2. The cut pst template.

    """

    # cut the data
    science_data_cut = science_data[:,
                                    cut_off_data:-cut_off_data,
                                    cut_off_data:-cut_off_data]

    psf_template_cut = psf_template[cut_off_psf:-cut_off_psf,
                                    cut_off_psf:-cut_off_psf]

    return science_data_cut,  psf_template_cut


def load_adi_data(
        hdf5_dataset: str,
        data_tag: str,
        psf_template_tag: str,
        para_tag="PARANG"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads ADI data stored as a hdf5 file. This function is needed to read
    the data and reproduce the results of the Apples with Apples paper.

    Args:
        hdf5_dataset: The path to the hdf5 file.
        data_tag: Tag of the science data in the hdf5 database.
        psf_template_tag: Tag of the PSF template in the hdf5 database.
        para_tag: Tag of the parallactic angles in the hdf5 database.

    Returns:
        1. The science data as a 3d numpy array.
        2. The parallactic angles as a 1d numpy array
        3. The PSF template as a 3d numpy array.
    """

    hdf5_file = h5py.File(hdf5_dataset, 'r')
    data = hdf5_file[data_tag][...]
    angles = np.deg2rad(hdf5_file[para_tag][...])
    psf_template_data = hdf5_file[psf_template_tag][...]
    hdf5_file.close()

    return data, angles, psf_template_data


def save_as_fits(
        data: np.ndarray,
        file_name: str
) -> None:
    """
    Saves data as .fits file.

    Args:
        data: The data to be saved.
        file_name: The filename of the fits file.
    """

    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])

    hdul.writeto(file_name)


def open_fits(file_name):
    """
    Opens a fits file as a numpy array.

    Args:
        file_name: Path to the fits file.

    Returns:
        Load data as numpy array.

    """

    with fits.open(file_name) as hdul:
        # pylint: disable=no-member
        data = deepcopy(hdul[0].data[...])
        del hdul[0].data
        del hdul

    return data
