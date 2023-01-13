"""
Simple helper functions to handle data
"""
import os
import h5py
from astropy.io import fits
import numpy as np
import time
import datetime
from copy import deepcopy


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


def load_adi_data(hdf5_dataset: str,
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
