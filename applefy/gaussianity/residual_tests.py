"""
Functions to test the statistics of residuals.
"""

import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score

from photutils import CircularAnnulus

from applefy.utils.general import center_subpixel
from applefy.utils.photometry import AperturePhotometryMode, \
    IterNoiseBySeparation


def extract_circular_annulus(
        separation,
        size_resolution_elements,
        frame_in,
        annulus_width=0.5):
    """
    Function to extract the pixel positions and values around a circular annulus
    at a given separation and annulus width.

    Args:
        separation: The separation from the center of the given frame in
            lambda / D (size_resolution_elements) at which the annulus is
            placed.
        size_resolution_elements: The size of the resolution elements i.e. the
            FWHM or lambda / D in pixel.
        frame_in: The input frame from which the pixel values are extracted.
        annulus_width: The width of the annulus radius in units of
            size_resolution_elements

    Returns: tuple with three values:
        1.) The pixel values in the annulus as 1D array
        2.) The 2D positions of the pixels in the annulus
        3.) A 2D image of the annulus mask used
    """

    frame_center = center_subpixel(frame_in)
    tmp_mask = CircularAnnulus(
        frame_center,
        size_resolution_elements * (separation - annulus_width),
        size_resolution_elements * (separation + annulus_width)).to_mask(
        'center')  # center mode returns a binary mask

    res_mask_img = tmp_mask.to_image(frame_in.shape)
    res_all_pixel = frame_in[res_mask_img != 0].flatten()
    tmp_positions = np.array(np.where(res_mask_img != 0)).T

    return res_all_pixel, tmp_positions, res_mask_img


def gaussian_r2(
        noise_samples,
        fit_method="least squares",
        return_fit=False):
    """
    Simple function to calculate how well the given noise samples can be
    explained by the normal distribution.

    Args:
        noise_samples: list of noise values
        fit_method: Method used to do the fit of the noise samples w.r.t the
            normal distribution. This is needed to compute the r2 metric.
            Option 1: "least squares" -  a simple least squares fit.
            Option 2: "theil sen" - TheilSenRegressor linear fit.
            More robust towards extreme samples.
        return_fit: If set to true the function will return the fit model and
            gaussian quantile points. This can be useful to plot Q-Q plots

    Returns: 1.) R2 - Coefficient of determination,
             2.) The linear model used for the fit (only if return_fit is True)
             3.) The gaussian_quantile points (only if return_fit is True)
    """

    gaussian_samples = stats.probplot(noise_samples)[0][0]

    if fit_method == "least squares":
        tmp_linear_model = linear_model.LinearRegression()
    elif fit_method == "theil sen":
        tmp_linear_model = linear_model.TheilSenRegressor()
    else:
        raise ValueError("Regression model unknown")

    tmp_linear_model.fit(np.sort(gaussian_samples).reshape(-1, 1),
                         np.sort(noise_samples).ravel())

    predictions = tmp_linear_model.predict(np.sort(
        gaussian_samples).reshape(-1, 1))

    r2 = r2_score(np.sort(noise_samples), predictions)

    if return_fit:
        return r2, tmp_linear_model, gaussian_samples

    return r2


def estimate_gaussian_r2(
        input_residual_frame,
        separation,
        size_resolution_elements,
        annulus_width=0.5,
        fit_method="least squares"):
    """
    Extracts pixel values inside a circular annulus around the center of the
    input residual frame and computes the r2 of the pixel values w.r.t. to
    the gaussian distribution.
    As pixel values in HCI residuals are usually not independent the result of
    the test should only be used as an indicator to a proof for gaussian
    residual noise.

    Args:
        input_residual_frame: The input frame on which the test is performed
            (2D array)
        separation: The separation from the center in lambda / D
            (size_resolution_elements) at which the annulus is placed and the
            noise elements are tested.
        size_resolution_elements: The size of the resolution elements i.e. the
            diameter of the FWHM in pixel.
        annulus_width: The width of the annulus radius in units of
            size_resolution_elements
        fit_method: Method used to do the fit of the noise samples w.r.t the
            normal distribution. This is needed to compute the r2 metric.
            Option 1: "least squares" -  a simple least squares fit.
            Option 2: "theil sen" - TheilSenRegressor linear fit.
            More robust towards extreme samples.

    Returns: R2
    """

    # 1.) Extract the pixel values on which the test is performed
    noise_elements, _, _ = extract_circular_annulus(
        separation=separation,
        size_resolution_elements=size_resolution_elements,
        frame_in=input_residual_frame,
        annulus_width=annulus_width)

    # 2.) compute the r2
    r2 = gaussian_r2(noise_samples=noise_elements,
                     fit_method=fit_method)

    return r2


def test_normality_shapiro_wilk(
        input_residual_frame,
        separation,
        size_resolution_elements,
        num_iterations,
        photometry_mode: AperturePhotometryMode):

    """
    Runs a Shapiro-Wilk test on photometry values at a given separation around
    the center of the input residual frame. The noise elements are sampled such
    that measurements are independent as required by the Shapiro-Wilk test.
    However, due to the small number of  residual elements at small separation
    the test has only very limited sensitivity.

    Args:
        input_residual_frame: The input frame on which the test is performed
            (2D array)
        separation: The separation from the center in lambda / D
            (size_resolution_elements) at which the noise photoementry is taken.
        size_resolution_elements: The size of the resolution elements i.e. the
            diameter of the FWHM in pixel.
        num_iterations: Number of different noise positions at which the
            Shapiro-Wilk test is evaluated.
        photometry_mode: An instance of AperturePhotometryMode which defines
            how the noise photometry is measured.

    Returns: The averaged p-values calculated for different noise positions:
        Statistic of the Shapiro-Wilk test (float),  p-value of the
        Shapiro-Wilk test (float)
    """

    # 1.) Create the iterator to extract the noise elements
    noise_iterator = IterNoiseBySeparation(
        residual=input_residual_frame,
        separation=separation * size_resolution_elements,
        psf_fwhm_radius=size_resolution_elements / 2.,
        num_iterations=num_iterations,
        photometry_mode=photometry_mode)

    # 2.) Loop over the noise elements and collect the p-values of the test
    p_values = []
    statistic_values = []

    for tmp_noise_samples in noise_iterator:
        p_values.append(stats.shapiro(tmp_noise_samples).pvalue)
        statistic_values.append(stats.shapiro(tmp_noise_samples).statistic)

    # 3.) Return the averaged values
    return np.mean(statistic_values),  np.mean(p_values)
