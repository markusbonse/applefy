"""
General functions needed across the test implementations
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


class TestInterface(ABC):
    """
    A general interface for statistical two sample tests. This interface
    guarantees that the code for the contrast curve calculations can be used
    with all metrics implemented in non_parametric.py, parametric.py and
    bootstrapping.py
    """

    def __init__(self,
                 num_cpus=1):
        """
        Constructor interface.

        Args:
            num_cpus: The number of CPU core that will be used in all subclass
                functions (e.g. bootstrapping).
        """
        self.m_num_cpus = num_cpus

    @abstractmethod
    def test_2samp(self,
                   planet_samples,
                   noise_samples):
        """
        Performs one (or multiple) two sample test given noise observations
        (X1, ...., Xn) and a planet observation (Y1,)
        with null hypothesis:
            mean(planet_distribution) == mean(noise_distribution)
        against the alternative:
            mean(planet_distribution) > mean(noise_distribution)

        Args:
            planet_samples: The planet sample containing the observation of the
                planet (Y1) as a float.
                In case multiple tests are performed the input can also be a
                list or 1D array ((Y1)1, (Y1)2, ...)
            noise_samples: List or 1D array of noise observations containing
                (float) values i.e. (X1, ...., Xn). In case multiple tests are
                performed the input can also be a list of lists / 1D arrays or
                a 2D array. ((X1, ...., Xn)1, (X1, ...., Xn)2, ... )

        Returns: The p-value (or list of p-values) of the test i.e. the FPF,
            the test statistic (or list of statistics)
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
                return

            if len(planet_samples.shape) > 1:
                raise ValueError("Input planet sample can only be a float or"
                                 " a 1D type (list or 1D array).")

        return

    @abstractmethod
    def constrain_planet(self,
                         noise_at_planet_pos,
                         noise_samples,
                         desired_confidence_fpf):
        """
        The inverse of test_2samp. Given noise observations (X1, ..., Xn-1) and
        a single noise observation (Xn) this function computes how much flux
        we have to add to Xn such that a test_2samp with noise (X1, ..., Xn-1)
        and planet signal (Y1 = Xn + added_flux) reaches a desired confidence
        level / p-value of the test. The added_flux is the flux a
        potential planet needs to add to the residual such that the test counts
        the observation as a detection. Similar to test_2samp this function
        also accepts lists as inputs to constrain multiple added_flux values at
        once.

        Args:
            noise_at_planet_pos: The noise observation (Xn) on top of which the
                planet is added (float). In case multiple values are constrained
                this can also be a list or 1D array of floats.
            noise_samples: List or 1D array of noise observations containing
                (float) values i.e. (X1, ...., Xn-1). In case multiple tests are
                performed the input can also be a list of lists / 1D arrays or
                a 2D array. ((X1, ...., Xn-1)1, (X1, ...., Xn-1)2, ... )
            desired_confidence_fpf: The desired confidence we want to reach as
                FPF. For example in case of a 5 sigma detection 2.87e-7.

        Returns: The flux we need to add to noise_at_planet_pos to reach the
            desired_confidence_fpf (float). In case of multiple test the output
            is a 1D array.
        """
        if not isinstance(noise_at_planet_pos, (np.floating, float)):
            if len(noise_at_planet_pos) != len(noise_samples):
                raise ValueError("If multiple tests are performed the input "
                                 "lists of noise_at_planet_pos and "
                                 "noise_samples need to have the same length. "
                                 "In case you intended to perform a single test"
                                 " make sure to pass noise_at_planet_pos as a"
                                 " float.")
        return


def fpf_2_gaussian_sigma(confidence_fpf):
    """
    Transforms a confidence level given as false-positive-fraction into a
    confidence level as sigma (in the gaussian sense).
    """

    return stats.norm.isf(confidence_fpf)


def gaussian_sigma_2_fpf(confidence_sigma):
    """
    Transforms a confidence level given as sigma (in the gaussian sense) into a
    confidence level as false-positive-fraction.
    """

    return stats.norm.sf(confidence_sigma)
