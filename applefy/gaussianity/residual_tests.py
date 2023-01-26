"""
Functions to evaluate the noise in high-contrast-imaging residuals. Note, none
of the tests implemented here can proof that the residual noise is Gaussian.
But they can provide useful insides whether the noise deviates for Gaussian.
"""
from typing import Union, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import TheilSenRegressor, LinearRegression
from sklearn.metrics import r2_score

from photutils.aperture import CircularAnnulus

from applefy.utils.positions import center_subpixel
from applefy.utils.photometry import AperturePhotometryMode, \
    IterNoiseBySeparation


def extract_circular_annulus(
        input_residual_frame: np.ndarray,
        separation: float,
        size_resolution_elements: float,
        annulus_width: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to extract the pixel positions and values around a circular annulus
    at a given separation form the center of an image.

    Args:
        input_residual_frame: The input frame from which the pixel values are
            extracted.
        separation: The separation from the center at which the annulus is
            placed (in units of size_resolution_elements).
        size_resolution_elements: The size of the resolution elements i.e. the
            diameter of the PSF FWHM in pixel.
        annulus_width: The width of the annulus radius in units of
            size_resolution_elements.

    Returns:
        1. The pixel values in the annulus extracted from the frame
        2. The 2D positions of the pixels in the annulus
        3. A 2D image of the annulus mask
    """

    frame_center = center_subpixel(input_residual_frame)
    tmp_mask = CircularAnnulus(
        frame_center,
        size_resolution_elements * (separation - annulus_width),
        size_resolution_elements * (separation + annulus_width)).to_mask(
        'center')  # center mode returns a binary mask

    res_mask_img = tmp_mask.to_image(input_residual_frame.shape)
    res_all_pixel = input_residual_frame[res_mask_img != 0].flatten()
    tmp_positions = np.array(np.where(res_mask_img != 0)).T

    return res_all_pixel, tmp_positions, res_mask_img


def gaussian_r2(
        noise_samples: np.ndarray,
        fit_method: str = "linear regression",
        return_fit: bool = False
) -> Union[float,
           Tuple[float,
                 Union[TheilSenRegressor, LinearRegression],
                 np.ndarray]]:
    """
    Simple function to calculate how well the given noise samples can be
    explained by the normal distribution.

    Args:
        noise_samples: Noise values to be compared against gaussian noise.
        fit_method: Method used to do the fit the noise samples w.r.t the
            normal distribution. This is needed to compute the r2 metric.

             Option 1: "linear regression" - A linear regression.

             Option 2: "theil sen" - TheilSenRegressor linear fit.
             More robust towards outliers.

        return_fit: If set to true the function will return the model fit and
            the gaussian quantile points. This can be useful to plot Q-Q plots.

    Returns:
        1. R2 - Coefficient of determination
        2. The linear model used for the fit (only if return_fit is True)
        3. The gaussian_quantile points (only if return_fit is True)

    """

    gaussian_samples = stats.probplot(noise_samples)[0][0]

    if fit_method == "linear regression":
        tmp_linear_model = LinearRegression()
    elif fit_method == "theil sen":
        tmp_linear_model = TheilSenRegressor()
    else:
        raise ValueError("Regression model unknown")

    tmp_linear_model.fit(np.sort(gaussian_samples).reshape(-1, 1),
                         np.sort(noise_samples).ravel())

    predictions = tmp_linear_model.predict(np.sort(
        gaussian_samples).reshape(-1, 1))

    r_2 = r2_score(np.sort(noise_samples), predictions)

    if return_fit:
        return r_2, tmp_linear_model, gaussian_samples

    return r_2


def estimate_gaussian_r2(
        input_residual_frame: np.ndarray,
        separation: float,
        size_resolution_elements: float,
        annulus_width: float = 0.5,
        fit_method: str = "linear regression"
) -> float:
    """
    Extracts pixel values inside a circular annulus around the center of the
    input_residual_frame and computes the r2 of the pixel values w.r.t. to
    the gaussian distribution.
    As neighbouring pixel values in HCI residuals are usually not independent
    the result of the test should only be used as an indicator against not a
    proof for gaussian residual noise.

    Args:
        input_residual_frame: The input input_residual_frame on which the test i
            s performed.
        separation: The separation from the center at which the annulus is
            placed and the noise gets extracted (in units of
            size_resolution_elements).
        size_resolution_elements: The size of the resolution elements i.e. the
            diameter of the PSF FWHM in pixel.
        annulus_width: The width of the annulus radius in units of
            size_resolution_elements
        fit_method: Method used to do the fit the noise samples w.r.t the
            normal distribution. This is needed to compute the r2 metric.

             Option 1: "linear regression" - A linear regression.

             Option 2: "theil sen" - TheilSenRegressor linear fit.
             More robust towards outliers.

    Returns:
        R2 - Coefficient of determination for the pixel values in the annulus.
    """

    # 1.) Extract the pixel values on which the test is performed
    noise_elements, _, _ = extract_circular_annulus(
        separation=separation,
        size_resolution_elements=size_resolution_elements,
        input_residual_frame=input_residual_frame,
        annulus_width=annulus_width)

    # 2.) compute the r2
    r_2 = gaussian_r2(noise_samples=noise_elements,
                      fit_method=fit_method)

    return r_2


def test_normality_shapiro_wilk(
        input_residual_frame: np.ndarray,
        separation: float,
        size_resolution_elements: float,
        num_rot_iter: int,
        photometry_mode: AperturePhotometryMode
) -> Tuple[float, float]:

    """
    Runs a Shapiro-Wilk test on photometry values at a given separation around
    the center of the input_residual_frame. The noise elements are sampled such
    that measurements are independent as required by the Shapiro-Wilk test.
    However, due to the small number of  residual elements at small separation
    the test has only very limited sensitivity. Further the test can never
    proof that the noise is Gaussian.

    Args:
        input_residual_frame: The frame on which the test is performed.
        separation: The separation from the center at which the noise
            photometry is taken (in units of size_resolution_elements).
        size_resolution_elements: The size of the resolution elements i.e. the
            diameter of the PSF FWHM in pixel.
        num_rot_iter: Number of different noise positions at which the
            Shapiro-Wilk test is evaluated. See
            `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
            for more information.
        photometry_mode: An instance of AperturePhotometryMode which defines
            how the noise photometry is measured.

    Returns:

        1. The average test statistic of the Shapiro-Wilk test over all
            num_rot_iter

        2. The average p-value of the Shapiro-Wilk test over all num_rot_iter
    """

    # 1.) Create the iterator to extract the noise elements
    noise_iterator = IterNoiseBySeparation(
        residual=input_residual_frame,
        separation=separation * size_resolution_elements,
        psf_fwhm_radius=size_resolution_elements / 2.,
        num_rot_iter=num_rot_iter,
        photometry_mode=photometry_mode)

    # 2.) Loop over the noise elements and collect the p-values of the test
    p_values = []
    statistic_values = []

    for tmp_noise_samples in noise_iterator:
        p_values.append(stats.shapiro(tmp_noise_samples).pvalue)
        statistic_values.append(stats.shapiro(tmp_noise_samples).statistic)

    # 3.) Return the averaged values
    return float(np.mean(statistic_values)),  float(np.mean(p_values))
