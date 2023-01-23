"""
Parametric tests for contrast and detection limit calculations
"""

import numpy as np
import multiprocessing
from scipy import stats
from numba import njit, set_num_threads

from applefy.statistics.general import TestInterface

################################################################################
# ------------------------------ Helper functions ------------------------------
################################################################################


@njit(parallel=True)
def _t_statistic_vectorized_numba(noise_samples,
                                  planet_samples):
    """
    Fast and parallel version of t_statistic_vectorized using numba. More
    information given in t_statistic_vectorized.
    """
    n = noise_samples.shape[1]
    m = 1

    res = []
    for i in range(noise_samples.shape[0]):
        x_bar_noise = noise_samples[i, :].mean()
        y_bar_planet = planet_samples[i]

        # np.sqrt(float(n) / float(n-1)) is the ddof = 1
        noise_std = noise_samples[i, :].std() * np.sqrt(
            1 / n + 1 / m) * np.sqrt(float(n) / float(n - 1))
        t = (y_bar_planet - x_bar_noise) / noise_std
        res.append(t)

    return np.array(res)


def t_statistic_vectorized(noise_samples,
                           planet_samples,
                           numba_parallel_threads=1):
    """
    Computes the test-statistic of the ttest / bootstrapping using vectorized
    code. Usually noise_samples and planet_samples contain a list of
    multiple samples. For every pair of noise and planet sample this function
    returns one t-value. In case more than 10e4 test values have to be computed
    a parallel implementation using numba can be used.

    Args:
        planet_samples: The planet sample containing the observation of the
            planet (Y1) as a float.
            In case multiple tests are performed the input can also be a
            list or 1D array ((Y1)1, (Y1)2, ...)
        noise_samples: List or 1D array of noise observations containing
            (float) values i.e. (X1, ...., Xn). In case multiple tests are
            performed the input can also be a list of lists / 1D arrays or
            a 2D array. ((X1, ...., Xn)1, (X1, ...., Xn)2, ... )
        numba_parallel_threads: The number of parallel threads used by numba.
            In case the function is used with multiprocessing this number
            should always be equal to the default of 1.

    Returns: float (in case of 1D input), list (in case of 2D input) of
        t-values / SNR
    """
    # make sure noise_samples is np.array
    noise_samples = np.array(noise_samples)

    if isinstance(planet_samples, (np.floating, float)):
        # Only a single test
        m = 1
        n = noise_samples.shape[0]
        x_bar_noise = np.mean(noise_samples)
        y_bar_planet = planet_samples
        noise = np.std(noise_samples, ddof=1) * np.sqrt(1 / n + 1 / m)

    else:
        # Multiple tests

        # in case a very large number of evaluations needs to be carried out
        # we can use a parallelized implementation using numba
        if (planet_samples.shape[0] > 10e4) and numba_parallel_threads != 1:

            set_num_threads(numba_parallel_threads)
            # use fast numba version
            results = _t_statistic_vectorized_numba(noise_samples,
                                                    planet_samples)
            set_num_threads(1)

            return results

        m = 1
        n = noise_samples.shape[1]
        x_bar_noise = np.mean(noise_samples, axis=1)
        y_bar_planet = planet_samples
        noise = np.std(noise_samples, axis=1, ddof=1) * np.sqrt(1 / n + 1 / m)

    tau = (y_bar_planet - x_bar_noise) / noise
    return tau

################################################################################
# ------------------------------ The Test --------------------------------------
################################################################################


class TTest(TestInterface):
    """
    A classical two sample ttest. Assumes iid Gaussian noise and tests for
    differences in means.
    """

    @classmethod
    def fpf2tau(cls,
                fpf,
                num_noise_values):
        """
        Computes the required value of tau (the test statistic) to get a
        confidence level of fpf. Takes into account the effect of the sample
        size by using the t-distribution.
        Accepts single value inputs as well as a list of fpf values.

        Args:
            fpf: Desired confidence level as FPF (float or list)
            num_noise_values: Number of noise observations. (int)

        Returns: The needed test statistic tau (float or list of floats)
        """
        n = num_noise_values
        m = 1

        tau = stats.t.isf(fpf, df=n + m - 2)
        return tau

    def tau2fpf(self,
                tau,
                num_noise_values):
        """
        Computed the confidence as fpf given the test statistic tau. Takes into
        account the effect of the sample size by using the t-distribution.
        Accepts single value inputs as well as a list of fpf values.

        Args:
            tau: The test statistic (float or list)
            num_noise_values: Number of noise observations. (int)

        Returns: The confidence / p-value / fpf of the test

        """
        n = num_noise_values
        m = 1

        if isinstance(tau, (float, np.floating)):
            fpf = stats.t.sf(tau, df=n + m - 2)

        # check if we can use multiprocessing for speedups
        elif len(tau) > 10e4:
            # split the tau values into 100 sub arrays and run them in parallel
            pool = multiprocessing.Pool(int(self.num_cpus))
            mp_results = pool.starmap(
                stats.t.sf,
                [(sub_array, m + n - 2) for sub_array in
                    np.array_split(tau, 100)])

            pool.close()

            fpf = np.concatenate(mp_results).flatten()
        else:
            fpf = stats.t.sf(tau, df=n + m - 2)

        return fpf

    def test_2samp(self,
                   planet_samples,
                   noise_samples):
        """
        Performs a two sample T-Test. This implementation is similar to the one
        in scipy but calculates the special case where only one planet sample
        is given. More information on the function can be found in the
        TestInterface.

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

        # run the super method to check if the shapes are correct
        super().test_2samp(planet_samples, noise_samples)

        # make sure noise_samples is np.array
        noise_samples = np.array(noise_samples)

        tau = t_statistic_vectorized(noise_samples,
                                     planet_samples,
                                     numba_parallel_threads=self.num_cpus)

        if isinstance(planet_samples, (np.floating, float)):
            num_noise_values = noise_samples.shape[0]
        else:
            num_noise_values = noise_samples.shape[1]

        p_values = self.tau2fpf(tau, num_noise_values)

        return p_values, tau

    def constrain_planet(self,
                         noise_at_planet_pos,
                         noise_samples,
                         desired_confidence_fpf):
        """
        The inverse of test_2samp. Given noise observations (X1, ..., Xn-1) and
        a single noise observation (Xn) this function computes how much flux
        we have to add to Xn such that a two sample ttest with noise
        (X1, ..., Xn-1) and planet signal (Y1 = Xn + added_flux) reaches a
        desired confidence level / p-value of the test. The added_flux is the
        flux a potential planet needs to add to the residual such that the test
        counts the observation as a detection. Similar to test_2samp this
        function also accepts lists as inputs to constrain multiple added_flux
        values at once.

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

        # run the super method to check if the shapes are correct
        super().constrain_planet(noise_at_planet_pos,
                                 noise_samples,
                                 desired_confidence_fpf)

        # make sure noise_samples is np.array
        noise_samples = np.array(noise_samples)

        # We reshape the noise_at_planet_pos to
        if isinstance(noise_at_planet_pos, (np.floating, float)):
            noise_at_planet_pos = np.array([noise_at_planet_pos, ])
            noise_samples = np.array([noise_samples, ])

        all_flux_needed = []
        for i in range(len(noise_at_planet_pos)):
            tmp_noise_at_planet_pos = noise_at_planet_pos[i]
            tmp_noise_samples = noise_samples[i]

            m = 1
            n = len(tmp_noise_samples)

            tmp_tau = self.fpf2tau(desired_confidence_fpf, n)
            tmp_sigma = np.std(tmp_noise_samples, ddof=1)
            tmp_noise = tmp_sigma * np.sqrt(1 / n + 1 / m)
            tmp_x_bar = np.mean(tmp_noise_samples)

            residual_flux_needed = tmp_tau * tmp_noise + tmp_x_bar
            residual_flux_needed -= tmp_noise_at_planet_pos
            all_flux_needed.append(residual_flux_needed)

        if len(all_flux_needed) == 1:
            return all_flux_needed[0]

        else:
            return np.array(all_flux_needed)
