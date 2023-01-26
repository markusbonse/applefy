"""
Util functions needed to compute contrast curves. The recommended way
to compute contrast curves is to use the class
:meth:`~applefy.detections.contrast.Contrast` and not the util functions.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

from applefy.statistics.general import TestInterface
from applefy.utils.photometry import IterNoiseBySeparation, \
    AperturePhotometryMode


def compute_contrast_curve(
        throughput_list: pd.DataFrame,
        stellar_flux: float,
        fp_residual: np.ndarray,
        confidence_level_fpf: float,
        statistical_test: TestInterface,
        psf_fwhm_radius: float,
        photometry_mode_noise: AperturePhotometryMode,
        num_rot_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    """
    Computes an analytic contrast curve given a confidence level and a
    statistical test. Analytic contrast curves are only applicable if used with
    linear post-processing techniques such as PCA.
    They can further lead to inaccurate results close to the star. For more
    advanced post-processing techniques use a contrast grid instead.

    Supports all statistical tests implemented in
    `Statistics <statistics.html>`_.

    Args:
        throughput_list: 1D pandas array of throughput values for every
            separation. (floats in range [0, 1]).
        stellar_flux: The stellar flux measured with
            :meth:`~applefy.utils.photometry.estimate_stellar_flux`. The mode
            used to estimate the stellar_flux has to be compatible with the
            photometry_mode_noise used here.
        fp_residual: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the noise statistics. This is usually the
            residual without fake planets.
        confidence_level_fpf: The confidence level associated with the
            contrast curve as false-positive fraction (FPF). Can also be a list
            of fpf values. In case a list is given the number of fpf values
            has to match the number of values in the throughput_list.
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
        photometry_mode_noise: An instance of AperturePhotometryMode which
            defines how the noise photometry is measured.
        num_rot_iter: Number of tests performed with different positions of
            the noise values. See
            `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
            for more information.

    Returns:
        1. median_contrast_curve - The median of all num_rot_iter contrast
        curves.

        2. mad_contrast_curve_error - The median absolute deviation of all
        num_rot_iter contrast curves.

        3. contrast_curves - A pandas DataFrame containing all individual
        num_rot_iter contrast curves calculated.

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
            num_rot_iter=num_rot_iter,
            photometry_mode=photometry_mode_noise,
            max_rotation=360)

        # Run evaluate contrast for num_rot_iter positions
        for rot_idx, tmp_observations in enumerate(noise_iterator):
            tmp_noise_at_planet = tmp_observations[0]
            tmp_noise_sample = tmp_observations[1:]

            residual_flux_needed = statistical_test.constrain_planet(
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
