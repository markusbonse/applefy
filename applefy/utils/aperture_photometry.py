"""
All util functions needed to estimate flux and do analysis with apertures
"""

import numpy as np
import warnings
from photutils import aperture_photometry, CircularAperture
from astropy.modeling import models, fitting
from abc import ABC, abstractmethod

from applefy.utils.aperture_positions import get_number_of_apertures,\
    estimate_aperture_positions, estimate_reference_positions
from applefy.utils.general import center_subpixel


def flux_ratio2mag(flux_ratios):
    """
    Convert a contrast given as a flux ratio to magnitudes.

    Args:
        flux_ratios: The contrast as a flux ratio; either as a single float or
            as a numpy array of floats.

    Returns:
        The contrast(s) in magnitudes.
    """
    return -np.log10(flux_ratios) * 2.5


def mag2flux_ratio(magnitudes):
    """
    Convert a contrast in magnitudes back to a flux ratio.

    Args:
        magnitudes: The contrast in magnitudes; either as a single
            float or as a numpy array of floats.

    Returns:
        The contrast(s) as a flux ratio.
    """
    return 10 ** (-magnitudes / 2.5)


class AperturePhotometryMode(object):
    """
    A capsule to manage the parameters of how photometry statistics are
    collected on residuals frames. The function is a wrapper of the parameters
    needed by the function utils.aperture_photometry.get_flux which is
    the foundation of many other functions.
    """

    def __init__(
            self,
            flux_mode,
            psf_fwhm_radius=None,
            search_area=None):
        """
        Constructor of the class. Takes the parameters needed by
        utils.aperture_photometry.get_aperture_flux and checks them for
        completeness.

        Args:
            flux_mode: The mode how photometry is measured.
                5 different modes are supported:
                1.) "AS" - Aperture sum: The aperture sum around the given
                    position inside one aperture_radius is calculated. Raises an
                    error if aperture_radius is not given.
                2.) "ASS" - Aperture sum + search: The aperture sum within a
                    radius of aperture_radius is calculated. The function
                    searches for the highest flux within one search_area. Raises
                    an error if aperture_radius or search_area are not given.
                3.) "P" - Pixel: The flux at the given pixel is calculated.
                    Interpolation is used to get sub-pixel values. P is
                    equivalent to "AS" with an aperture diameter of one pixel
                    (i.e. Circular Aperture).
                4.) "F" - Fit: A 2D gaussian is fit at the given position. The
                    function returns its amplitude.
                5.) "FS" - Fit + search:  A 2D gaussian is fit around the given
                    position within one search_area. The function returns its
                    amplitude and final position of the fit.
                6.) "PG" - Pixel on Grid: Returns value of the pixel with the
                    closest position. Not compatible with mode "P" since the
                    pixel grid is not circular. The mode "PG" should only be
                    used to test temporal similarities.
            psf_fwhm_radius: Needed for modes "AS" and "ASS". Gives the aperture
                radius for the circular aperture used to calculate the summed up
                flux. [pixel] (float)
            search_area: Needed for modes "ASS" and "FS". Gives the search are
                which is considered to find the highest flux. [pixel] (float)
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

    def check_compatible(self, other_aperture_mode):
        """
        Check if the mode is compatible with a second given mode. Returns False
            if the modes are not compatible

        Args:
            other_aperture_mode: A second instance of AperturePhotometryMode
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


def get_flux(
        frame,
        position,
        photometry_mode: AperturePhotometryMode):
    """
    Function to estimate the flux at and / or around a given position in frame.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the aperture photometry.
        position: A tuple `(x, y)` specifying the position at which we estimate
            the photometry.
        photometry_mode: An instance of AperturePhotometryMode which defines how
            the flux is measured at the given position.

    Returns: A tuple ((final_pos_x, final_pos_y), estimated_flux)
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

    elif photometry_mode.flux_mode in ["F", "FS"]:
        # Modes with Gaussian fit

        # Define the grid for the fit
        x = np.arange(frame.shape[0])
        y = np.arange(frame.shape[1])
        x, y = np.meshgrid(x, y)

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
            gaussian_model = fit_p(gaussian_model, x, y, np.nan_to_num(frame))

        position = (gaussian_model.x_mean.value, gaussian_model.y_mean.value)

        # estimate the flux on the fit frame
        # We can not use the amplitude directly as it is not comparable with
        # noise values which are e.g. estimated in pixel mode
        fit_result = gaussian_model(x, y)

        tmp_aperture = CircularAperture(positions=position,
                                        r=0.5)

        photometry_table = aperture_photometry(fit_result,
                                               tmp_aperture,
                                               method='exact')

        flux = photometry_table["aperture_sum"][0]

        return position, flux

    else:

        # cast clamp the position to the next pixel
        new_position = tuple(np.round(np.array(position)))
        flux = frame[int(new_position[0]), int(new_position[1])]

        return new_position, flux


class IterNoise(ABC):
    """
    Abstract interface for IterNoiseBySeparation and IterNoiseForPlanet
    """

    def __init__(
            self,
            residual,
            separation,
            psf_fwhm_radius,
            num_iterations,
            photometry_mode: AperturePhotometryMode,
            max_rotation):

        self.residual = residual
        self.separation = separation
        self.psf_fwhm_radius = psf_fwhm_radius
        self.num_iterations = num_iterations
        self.center = center_subpixel(residual)
        self.photometry_mode = photometry_mode
        self.max_rotation = max_rotation

    @abstractmethod
    def _calc_noise_positions(self, tmp_angle_offset):
        pass

    def _calc_angle_offsets(self):
        if self.max_rotation is None:
            tmp_max_rotation = 360 / get_number_of_apertures(
                self.separation,
                self.psf_fwhm_radius)
        else:
            tmp_max_rotation = self.max_rotation

        return np.linspace(0, np.deg2rad(tmp_max_rotation),
                           self.num_iterations)

    def __iter__(self):
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
    rotation on the estimation of contrast.
    """

    def __init__(
            self,
            residual,
            separation,
            psf_fwhm_radius,
            num_iterations,
            photometry_mode: AperturePhotometryMode,
            max_rotation=360):
        """
        Constructor of the IterNoiseBySeparation.

        Args:
            residual: A 2D numpy array of shape `(width, height)` containing
                the data on which to run the aperture photometry. This is
                usually the residual without fake planets.
            separation: separation from the center [pixel] (float)
            psf_fwhm_radius: The separation radius between noise elements.
                [pixel] (float)
            num_iterations: Number of offset angles to be evaluated
            photometry_mode: An instance of AperturePhotometryMode which defines
                how the flux is measured at the calculated position.
            max_rotation: The maximum amount of rotation considered (deg). If
                None the maximum amount of rotation is computed based on the
                separation.
        """
        super().__init__(residual=residual,
                         separation=separation,
                         psf_fwhm_radius=psf_fwhm_radius,
                         num_iterations=num_iterations,
                         photometry_mode=photometry_mode,
                         max_rotation=max_rotation)

    def _calc_noise_positions(self,
                              tmp_angle_offset):
        return estimate_aperture_positions(
            self.separation,
            self.center,
            self.psf_fwhm_radius,
            angle_offset=tmp_angle_offset)


class IterNoiseForPlanet(IterNoise):
    """
    An iterator that allows to sample noise photometry for different reference
    positions relative to a given planet position. This is needed to average out
    the effect of rotation on the contrast.
    """

    def __init__(
            self,
            residual,
            planet_position,
            safety_margin,
            psf_fwhm_radius,
            num_iterations,
            photometry_mode: AperturePhotometryMode,
            max_rotation=None):
        """
        Constructor of the IterNoiseForPlanet

        Args:
            residual: A 2D numpy array of shape `(width, height)` containing
                the data on which to run the aperture photometry. This is
                usually the residual without fake planets.
            planet_position: The position of the planet. Is used to determine
                the separation and exclusion region. [tuple - (x, y) pixel]
            safety_margin: separation from the planet_position in which
                apertures are ignored. If None apertures within two
                aperture_radius are ignored
            psf_fwhm_radius: The separation radius between noise elements.
                [pixel] (float)
            num_iterations: Number of offset angles to be evaluated
            photometry_mode: An instance of AperturePhotometryMode which defines
                how the flux is measured at the calculated position.
            max_rotation: The maximum amount of rotation considered (deg). If
                None the maximum amount of rotation is computed based on the
                separation.
        """

        super().__init__(
            residual=residual,
            separation=None,
            psf_fwhm_radius=psf_fwhm_radius,
            num_iterations=num_iterations,
            photometry_mode=photometry_mode,
            max_rotation=max_rotation)

        self.planet_position = planet_position
        self.safety_margin = safety_margin

        self.separation = np.linalg.norm(
            np.array(self.planet_position) -
            np.array(self.center))

    def _calc_noise_positions(self, tmp_angle_offset):

        return estimate_reference_positions(
            planet_position=self.planet_position,
            center=self.center,
            psf_fwhm_radius=self.psf_fwhm_radius,
            angle_offset=tmp_angle_offset,
            safety_margin=self.safety_margin)
