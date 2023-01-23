"""
Functions and Tools to generate fake planet data sets as input for any
post-processing method.
"""
import numpy as np
from scipy.ndimage import shift


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
