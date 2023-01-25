"""
Simple helper functions to handle data
"""
import os
import json
import h5py
import warnings
from pathlib import Path
from astropy.io import fits
import numpy as np
from copy import deepcopy


def create_checkpoint_folders(checkpoint_dir):

    # if no experiment_root_dir is given we don't save results
    if checkpoint_dir is None:
        return

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
        config_dir,
        method_dir):

    collected_result_file = []

    # find all config files
    config_files = dict(collect_all_data_setup_configs(config_dir))

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


def collect_all_data_setup_configs(data_setups_dir):
    """
    Simple function which looks for all auto generated contrast map config
    files in one directory

    Args:
        data_setups_dir: The directory to be browsed by the method

    Returns: a list of tuples (job_id, file path)

    """
    # 1.) Collect all jobs to be run
    all_datasets_configs = []
    for tmp_file in sorted(os.listdir(data_setups_dir)):
        if not tmp_file.startswith("exp_"):
            continue

        tmp_id = tmp_file.split(".")[0].split("_")[-1]
        all_datasets_configs.append(
            (tmp_id, os.path.join(data_setups_dir, tmp_file)))

    return all_datasets_configs


def read_apples_with_apples_root():
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


def cut_and_stack(psf_template: np.array,
                  science_data: np.array,
                  par_angles: np.array,
                  config: dict):

    cut_off_psf = config["psf_template_cut"]
    cut_off_data = config["cut_off"]

    # cut the data
    science_data_cut = science_data[:,
                                    cut_off_data:-cut_off_data,
                                    cut_off_data:-cut_off_data]

    psf_template_cut = psf_template[cut_off_psf:-cut_off_psf,
                                    cut_off_psf:-cut_off_psf]

    return science_data_cut, par_angles, psf_template_cut


def load_adi_data(
        hdf5_dataset: str,
        data_tag: str,
        psf_template_tag: str,
        para_tag="PARANG"):
    """
    Function to load ADI data from hdf5 files
    :param hdf5_dataset: The path to the hdf5 file
    :param data_tag: Tag of the science data
    :param psf_template_tag: Tag of the PSF template
    :param para_tag: Tag of the parallactic angles
    :return: Tuple (Science, parang, template)
    """
    hdf5_file = h5py.File(hdf5_dataset, 'r')
    data = hdf5_file[data_tag][...]
    angles = np.deg2rad(hdf5_file[para_tag][...])
    psf_template_data = hdf5_file[psf_template_tag][...]
    hdf5_file.close()

    return data, angles, psf_template_data


def save_as_fits(data, file_name):

    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])

    hdul.writeto(file_name)


def open_fits(file_name):
    """
    Opens a fits file as numpy array
    :param file_name: Path to the fits file
    :return: the loaded data
    """
    with fits.open(file_name) as hdul:
        data = deepcopy(hdul[0].data[...])
        del hdul[0].data
        del hdul

    return data


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


def read_fake_planet_results(result_files):
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
