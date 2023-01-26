"""
All util functions needed to estimate flux and do analysis with apertures and
pixel values.
"""
import warnings
from abc import ABC, abstractmethod

from typing import Tuple, List, Union, Optional

import numpy as np
from photutils.aperture import aperture_photometry, CircularAperture
from astropy.modeling import models, fitting

from applefy.utils.positions import get_number_of_apertures,\
    estimate_noise_positions, estimate_reference_positions, center_subpixel


def flux_ratio2mag(
        flux_ratios: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert a contrast given as a flux ratio to magnitudes.

    Args:
        flux_ratios: The contrast as a flux ratio; either as a single float or
            as a numpy array of floats.

    Returns:
        The contrast(s) in magnitudes.
    """
    return -np.log10(flux_ratios) * 2.5


def mag2flux_ratio(
        magnitudes:Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert a contrast in magnitudes back to a flux ratio.

    Args:
        magnitudes: The contrast in magnitudes; either as a single
            float or as a numpy array of floats.

    Returns:
        The contrast(s) as a flux ratio.
    """
    return 10 ** (-magnitudes / 2.5)


class AperturePhotometryMode:
    """
    A capsule to manage the parameters of how photometry is calculated on
    residuals frames. The function is a wrapper of the parameters
    needed by the function :meth:`~get_flux` which is the foundation
    of many other functions.
    """

    def __init__(
            self,
            flux_mode: str,
            psf_fwhm_radius: Optional[float] = None,
            search_area: Optional[float] = None):
        """
        Constructor of the class. Takes the parameters needed by
        :meth:`~get_flux` and checks them for completeness.

        Args:
            flux_mode: The mode how photometry is measured.
                5 different modes are supported:

                    1. "AS" (Aperture sum) - The aperture sum around the given
                    position inside one psf_fwhm_radius is calculated. Raises an
                    error if psf_fwhm_radius is not given.

                    2. "ASS" (Aperture sum + search) - The aperture sum within a
                    radius of psf_fwhm_radius is calculated. The function
                    searches for the highest flux within one search_area. Raises
                    an error if psf_fwhm_radius or search_area are not given.

                    3. "P" (Pixel) - The flux at the given pixel is calculated.
                    Interpolation is used to get sub-pixel values. P is
                    equivalent to "AS" with an aperture diameter of one pixel
                    (i.e. Circular Aperture).

                    4. "F" (Fit) - A 2D gaussian is fit at the given position.
                    The function returns the same as "P" but based on the fit.

                    5. "FS" (Fit + search) - A 2D gaussian is fit around the
                    given position within one search_area. The function returns
                    the same as "P" but based on the fit.

                    6. "PG" (Pixel on Grid) -  Returns value of the pixel with
                    the closest position. Not compatible with mode "P" since the
                    pixel grid is not circular. The mode "PG" should only be
                    used to test temporal similarities.

            psf_fwhm_radius: Needed for modes "AS" and "ASS". Gives the aperture
                radius for the circular aperture used to calculate the summed
                flux. [pixel]
            search_area: Needed for modes "ASS" and "FS". Gives the search are
                which is considered to find the highest flux. [pixel]
        """

        # Check if the given setup makes sense
        if flux_mode not in ["AS", "ASS", "P", "F", "FS", "PG"]:
            raise ValueError("Mode " + str(flux_mode) + " not supported")

        if flux_mode in ["AS", "ASS"] and psf_fwhm_radius is None:
            raise ValueError("Modes AS and ASS need an aperture_radius")

        if flux_mode in ["ASS", "FS"] and search_area is None:
            raise ValueError("Modes ASS and FS need a search_area")

        self.flux_mode = flux_mode
        self.aperture_radius = psf_fwhm_radius
        self.search_area = search_area

    def check_compatible(
            self,
            other_aperture_mode: "AperturePhotometryMode"
    ) -> bool:
        """
        Check if the mode is compatible with a second given mode. Returns False
        if the modes are not compatible.

        Args:
            other_aperture_mode: A second instance of AperturePhotometryMode.

        Returns:
            Returns False if the modes are not compatible, True if they are.
        """

        if not isinstance(other_aperture_mode, AperturePhotometryMode):
            raise ValueError("The given other_aperture_mode is not an instance"
                             "of AperturePhotometryMode. Compatibility check "
                             "not possible.")
        # If self.flux_mode is AS or ASS the other_aperture_mode has to be
        # AS or ASS as well
        if self.flux_mode in ["AS", "ASS"]:
            return other_aperture_mode.flux_mode in ["AS", "ASS"]

        # If self.flux_mode is P, F or FS the other_aperture_mode has to be
        # P or F as FS
        if self.flux_mode in ["P", "FS", "F"]:
            return other_aperture_mode.flux_mode in ["P", "FS", "F"]

        # The mode PG has a square aperture and is only compatible with itself
        if self.flux_mode == "PG":
            return other_aperture_mode.flux_mode in ["PG", ]

        return False


def get_flux(
        frame: np.ndarray,
        position: Tuple[float, float],
        photometry_mode: AperturePhotometryMode
) -> Tuple[Tuple[float, float], float]:
    """
    Function to estimate the flux at / or around a given position in the
    frame.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the photometry.
        position: A tuple `(x, y)` specifying the position at which we estimate
            the photometry.
        photometry_mode: An instance of AperturePhotometryMode which defines how
            the flux is measured at the given position.

    Returns:
        A tuple with ((final_pos_x, final_pos_y), estimated_flux). final_pos
        contain the position of maximum flux in case the photometry_mode uses
        search.
    """

    if photometry_mode.flux_mode in ["AS", "ASS", "P"]:
        # Modes based on apertures

        if photometry_mode.flux_mode == "ASS":
            offset_range = np.linspace(-photometry_mode.search_area,
                                       photometry_mode.search_area, 5)
            new_positions = np.array(np.meshgrid(
                offset_range + position[0],
                offset_range + position[1])).reshape(2, -1).T
        else:
            new_positions = position

        # A pixel is handled as an aperture with one pixel diameter
        if photometry_mode.flux_mode == "P":
            aperture_radius = 0.5
        else:
            aperture_radius = photometry_mode.aperture_radius

        tmp_apertures = CircularAperture(positions=new_positions,
                                         r=aperture_radius)

        photometry_table = aperture_photometry(
            frame,
            tmp_apertures,
            method='exact')

        best_idx = np.argmax(photometry_table["aperture_sum"])
        best_aperture_sum = photometry_table["aperture_sum"][best_idx]
        best_position = (
            photometry_table["xcenter", "ycenter"][best_idx][0].value,
            photometry_table["xcenter", "ycenter"][best_idx][1].value)

        return best_position, best_aperture_sum

    if photometry_mode.flux_mode in ["F", "FS"]:
        # Modes with Gaussian fit

        # Define the grid for the fit
        x_range = np.arange(frame.shape[0])
        y_range = np.arange(frame.shape[1])
        x_range, y_range = np.meshgrid(x_range, y_range)

        # Create a new Gaussian2D object
        gaussian_model = models.Gaussian2D(x_mean=position[0],
                                           y_mean=position[1])

        # Enforce symmetry:
        # Tie standard deviation parameters to same value
        def tie_stddev(gaussian_model_local):
            return gaussian_model_local.y_stddev

        gaussian_model.x_stddev.tied = tie_stddev

        # Optional: Fix x and y position
        if photometry_mode.flux_mode == "F":
            gaussian_model.x_mean.fixed = True
            gaussian_model.y_mean.fixed = True
        else:
            gaussian_model.x_mean.min = \
                position[0] - photometry_mode.search_area
            gaussian_model.x_mean.max = \
                position[0] + photometry_mode.search_area
            gaussian_model.y_mean.min = \
                position[1] - photometry_mode.search_area
            gaussian_model.y_mean.max = \
                position[1] + photometry_mode.search_area

        # Fit the model to the data
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gaussian_model = fit_p(
                gaussian_model,
                x_range,
                y_range,
                np.nan_to_num(frame))

        position = (gaussian_model.x_mean.value, gaussian_model.y_mean.value)

        # estimate the flux on the fit input_residual_frame
        # We can not use the amplitude directly as it is not comparable with
        # noise values which are e.g. estimated in pixel mode
        fit_result = gaussian_model(x_range, y_range)

        tmp_aperture = CircularAperture(positions=position,
                                        r=0.5)

        photometry_table = aperture_photometry(fit_result,
                                               tmp_aperture,
                                               method='exact')

        flux = photometry_table["aperture_sum"][0]

        return position, flux

    # cast clamp the position to the next pixel
    new_position = tuple(np.round(np.array(position)))
    flux = frame[int(new_position[0]), int(new_position[1])]

    return new_position, flux


class IterNoise(ABC):
    """
    Abstract interface for :meth:`~IterNoiseBySeparation` and
    :meth:`~IterNoiseForPlanet`.
    """

    def __init__(
            self,
            residual: np.ndarray,
            separation: float,
            psf_fwhm_radius: float,
            num_rot_iter: int,
            photometry_mode: AperturePhotometryMode,
            max_rotation: Optional[float]):
        """
        Constructor of IterNoise.

        Args:
            residual: A 2D numpy array of shape `(width, height)` containing
                the data on which to run the photometry estimation.
            separation: Separation from the center [pixel].
            psf_fwhm_radius: The separation radius between noise elements
                [pixel].
            num_rot_iter: Number of tests performed with different positions of
                the noise values. See
                `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
                for more information.
            photometry_mode: An instance of AperturePhotometryMode which defines
                how the flux is measured at the calculated position.
            max_rotation: The maximum amount of rotation considered (deg). If
                None the maximum amount of rotation is computed based on the
                separation.
        """

        self.residual = residual
        self.separation = separation
        self.psf_fwhm_radius = psf_fwhm_radius
        self.num_rot_iter = num_rot_iter
        self.center = center_subpixel(residual)
        self.photometry_mode = photometry_mode
        self.max_rotation = max_rotation

    @abstractmethod
    def _calc_noise_positions(
            self,
            tmp_angle_offset: float
    ) -> List[Tuple[float, float, float, float]]:
        """
        Calculate the noise positions given a current angle_offset.

        Args:
            tmp_angle_offset: The angle_offset in (deg).

        Returns:
            the noise positions.
        """

        return [(-1, -1, -1, -1),]

    def _calc_angle_offsets(self) -> np.ndarray:
        """
        Calculates by how much we have to rotate the noise positions at a given
        separation until they overlap again the initial ones. Uses num_rot_iter
        to create a numpy array of angles offsets used in __iter__.

        Returns:
            numpy array of angles offsets
        """
        if self.max_rotation is None:
            tmp_max_rotation = 360 / get_number_of_apertures(
                self.separation,
                self.psf_fwhm_radius)
        else:
            tmp_max_rotation = self.max_rotation

        return np.linspace(0, np.deg2rad(tmp_max_rotation),
                           self.num_rot_iter)

    def __iter__(self) -> List[float]:
        """
        Allows to iterate over different noise positions and extract their flux.

        Returns:
            A list of flux values extracted at the current positions.
        """
        angle_offsets = self._calc_angle_offsets()

        for tmp_angle_offset in angle_offsets:
            # 1.) sample new positions
            tmp_positions = self._calc_noise_positions(tmp_angle_offset)

            # 2.) calculate photometry
            all_flux_values = []
            for tmp_position in tmp_positions:
                tmp_flux = get_flux(
                    self.residual,
                    tmp_position[:2],  # the first two vales are the x and y pos
                    photometry_mode=self.photometry_mode)

                # tmp_flux contains the final position and flux
                all_flux_values.append(tmp_flux[1])

            yield all_flux_values


class IterNoiseBySeparation(IterNoise):
    """
    An iterator that allows to sample noise photometry for different reference
    positions (angle_offsets). This is needed to average out the effect of
    rotation on the contrast. See
    `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
    for more information.
    """

    def __init__(
            self,
            residual: np.ndarray,
            separation: float,
            psf_fwhm_radius: float,
            num_rot_iter: int,
            photometry_mode: AperturePhotometryMode,
            max_rotation: float = 360):
        """
        Constructor of the IterNoiseBySeparation.

        Args:
            residual: A 2D numpy array of shape `(width, height)` containing
                the data on which to run the photometry estimation.
            separation: Separation from the center [pixel].
            psf_fwhm_radius: The separation radius between noise elements
                [pixel].
            num_rot_iter: Number of tests performed with different positions of
                the noise values.
            photometry_mode: An instance of AperturePhotometryMode which defines
                how the flux is measured at the calculated position.
            max_rotation: The maximum amount of rotation considered (deg). If
                None the maximum amount of rotation is computed based on the
                separation.
        """
        super().__init__(residual=residual,
                         separation=separation,
                         psf_fwhm_radius=psf_fwhm_radius,
                         num_rot_iter=num_rot_iter,
                         photometry_mode=photometry_mode,
                         max_rotation=max_rotation)

    def _calc_noise_positions(
            self,
            tmp_angle_offset: float
    ) -> List[Tuple[float, float, float, float]]:
        """
        Calculate the noise positions given a current angle_offset.

        Args:
            tmp_angle_offset: The angle_offset in (deg).

        Returns:
            the noise positions.
        """

        return estimate_noise_positions(
            self.separation,
            self.center,
            self.psf_fwhm_radius,
            angle_offset=tmp_angle_offset)


class IterNoiseForPlanet(IterNoise):
    """
    An iterator that allows to sample noise photometry for different reference
    positions relative to a given planet position. This is needed to average out
    the effect of rotation on the detection uncertainty. See
    `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
    for more information.
    """

    def __init__(
            self,
            residual: np.ndarray,
            planet_position: Tuple[float, float],
            safety_margin: float,
            psf_fwhm_radius: float,
            num_rot_iter: int,
            photometry_mode: AperturePhotometryMode,
            max_rotation: Optional[float] = None):
        """
        Constructor of the IterNoiseForPlanet

        Args:
            residual: A 2D numpy array of shape `(width, height)` containing
                the data on which to run the photometry estimation.
            planet_position: The position of the planet. Is used to determine
                the separation and exclusion region. [pixel]
            safety_margin: separation from the planet_position in which
                apertures are ignored. If None positions within two
                psf_fwhm_radius are ignored
            psf_fwhm_radius: The separation radius between noise elements.
                [pixel]
            num_rot_iter: Number of tests performed with different positions of
                the noise values.
            photometry_mode: An instance of AperturePhotometryMode which defines
                how the flux is measured at the calculated position.
            max_rotation: The maximum amount of rotation considered (deg). If
                None the maximum amount of rotation is computed based on the
                separation.
        """

        super().__init__(
            residual=residual,
            separation=0,
            psf_fwhm_radius=psf_fwhm_radius,
            num_rot_iter=num_rot_iter,
            photometry_mode=photometry_mode,
            max_rotation=max_rotation)

        self.planet_position = planet_position
        self.safety_margin = safety_margin

        self.separation = np.linalg.norm(
            np.array(self.planet_position) -
            np.array(self.center))

    def _calc_noise_positions(
            self,
            tmp_angle_offset: float
    ) -> List[Tuple[float, float, float, float]]:
        """
        Calculate the noise positions given a current angle_offset.

        Args:
            tmp_angle_offset: The angle_offset in (deg).

        Returns:
            the noise positions.
        """

        return estimate_reference_positions(
            planet_position=self.planet_position,
            center=self.center,
            psf_fwhm_radius=self.psf_fwhm_radius,
            angle_offset=tmp_angle_offset,
            safety_margin=self.safety_margin)


def estimate_stellar_flux(
        psf_template: np.ndarray,
        dit_science: float,
        dit_psf_template: float,
        photometry_mode: AperturePhotometryMode,
        scaling_factor: float = 1.0
) -> float:
    """
    Function to estimate the normalized flux of the star given an unsaturated
    PSF image.

    Args:
        psf_template: 2D array of the unsaturated PSF
        dit_science: Integration time of the science frames
        dit_psf_template: Integration time of unsaturated PSF
        scaling_factor: A scaling factor to account for ND filters.
        photometry_mode: An instance of AperturePhotometryMode which defines
            how the stellar flux is measured.

    Returns:
        The stellar flux

    """
    # Account for the integration time of the psf template
    integration_time_factor = dit_science / dit_psf_template * scaling_factor
    psf_norm = psf_template * integration_time_factor

    center = center_subpixel(psf_norm)

    _, flux = get_flux(psf_norm, center,
                       photometry_mode=photometry_mode)

    return flux
