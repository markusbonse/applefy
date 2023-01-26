"""
The interface needed to make all statistical tests compatible with the contrast
curve calculation.
"""

from typing import Union, List, Tuple
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


class TestInterface(ABC):
    """
    A general interface for two sample tests. This interface guarantees that the
    code for the contrast curve / grid calculations can be used with all test
    implemented.
    """

    def __init__(
            self,
            num_cpus: int = 1):
        """
        The tests might use multiprocessing.

        Args:
            num_cpus: The number of CPU cores that will be used in all tests
                (e.g. to run the bootstrapping).
        """
        self.num_cpus = num_cpus

    @abstractmethod
    def test_2samp(
            self,
            planet_samples: Union[float, np.ndarray],
            noise_samples: Union[List[float], np.ndarray]
    ) -> Union[Tuple[float, float],
               Tuple[np.ndarray, np.ndarray]]:
        r"""
        Performs one (or multiple) two sample test given noise observations
        :math:`(X_1, ..., X_n)` and a planet observation :math:`Y_1,`
        with null hypothesis:

        :math:`H_0: \mu_X = \mu_Y`

        against the alternative:

        :math:`H_1: \mu_X < \mu_Y`

        Args:
            planet_samples: The planet sample containing the observation of the
                planet :math:`Y_1` as a float.
                In case multiple tests are performed the input can also be a
                list or 1D array :math:`((Y_1)_1, (Y_1)_2, ...)`.
            noise_samples: The noise observations containing
                :math:`(X_1, ..., X_n)`. In case multiple tests are
                performed the input can also be a list of lists / 1D arrays or
                a 2D array:
                :math:`(X_1, ..., X_n)_1, (X_1, ..., X_n)_2, ...`.

        Returns:
            1. The p-value (or a 1d array of p-values) of the test i.e. the FPF.
            The interface returns -1.

            2. The test statistic :math:`T_{obs},` (or a 1d array of
            the statistics). The interface returns -1.
        """
        # Perform some simple tests on the inputs
        if not isinstance(planet_samples, (np.floating, float)):
            if len(planet_samples) != len(noise_samples):
                raise ValueError("If multiple tests are performed the input "
                                 "lists of planet_samples and noise_samples "
                                 "need to have the same length. In case you "
                                 "intended to perform a single test make sure "
                                 "to pass planet_samples as a float.")

            if isinstance(planet_samples, list):
                return -1, -1

            if len(planet_samples.shape) > 1:
                raise ValueError("Input planet sample can only be a float or"
                                 " a 1D type (list or 1D array).")

        return -1, -1

    @abstractmethod
    def constrain_planet(
            self,
            noise_at_planet_pos: Union[float, List[float], np.ndarray],
            noise_samples: Union[List[float], np.ndarray],
            desired_confidence_fpf: float
    ) -> Union[float, np.ndarray]:
        """
        The inverse of test_2samp. Given noise observations
        :math:`(X_1, ..., X_{n-1})` and a single noise observation
        :math:`X_n` this function computes how much flux we have to add to
        :math:`X_n` such that a test_2samp with noise
        :math:`(X_1, ..., X_{n-1})` and planet signal
        :math:`Y_1 = X_n + f` rejects the null hypothesis i.e. reaches
        the desired confidence level. The added flux :math:`f` is the flux a
        potential planet needs to have such that we would count is as a
        detection (assuming we observe it together with the noise at
        :math:`X_n`). Similar to test_2samp this function also accepts lists as
        inputs to constrain multiple added_flux values at the same time.

        Args:
            noise_at_planet_pos: The noise observation :math:`X_n` on top of
                which the planet is added. In case multiple values are
                constrained at the same time this can also be a list or 1D
                array of floats.
            noise_samples: List or 1D array of noise observations containing
                (:math:`(X_1, ..., X_{n-1})`. In case multiple tests are
                performed the input can also be a list of lists / 1D arrays or
                a 2D array:
                :math:`(X_1, ..., X_{n-1})_1, (X_1, ..., X_{n-1})_2, ...`.
            desired_confidence_fpf: The desired confidence we want to reach as
                FPF. For example in case of a 5 sigma detection 2.87e-7.

        Returns:
            The flux we need to add to noise_at_planet_pos to reach the
            desired_confidence_fpf :math:`f`. In case of multiple test the
            output is a 1D array. The interface returns -1.
        """
        if not isinstance(noise_at_planet_pos, (np.floating, float)):
            if len(noise_at_planet_pos) != len(noise_samples):
                raise ValueError("If multiple tests are performed the input "
                                 "lists of noise_at_planet_pos and "
                                 "noise_samples need to have the same length. "
                                 "In case you intended to perform a single test"
                                 " make sure to pass noise_at_planet_pos as a"
                                 " float.")
        return -1


def fpf_2_gaussian_sigma(
        confidence_fpf: Union[float, np.ndarray]
) -> float:
    r"""
    Transforms a confidence level given as false-positive-fraction / p-value
    into a confidence level in the gaussian sense :math:`\sigma_\mathcal{N}`.

    Args:
        confidence_fpf: The FPF we want to translate.

    Returns:
        :math:`\sigma_\mathcal{N}`

    """

    return stats.norm.isf(confidence_fpf)


def gaussian_sigma_2_fpf(
        confidence_sigma: Union[float, np.ndarray]
) -> float:
    r"""
    Transforms a confidence level given as :math:`\sigma_\mathcal{N}` into a
    confidence level as false-positive-fraction.

    Args:
        confidence_sigma: The FPF as :math:`\sigma_\mathcal{N}`.

    Returns:
        p-value / false-positive-fraction

    """

    return stats.norm.sf(confidence_sigma)
