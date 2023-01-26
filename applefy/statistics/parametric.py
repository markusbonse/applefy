"""
Parametric tests for contrast and detection limit calculations.
"""
from typing import Union, List, Tuple

import multiprocessing
import numpy as np
from scipy import stats
from numba import njit, set_num_threads

from applefy.statistics.general import TestInterface

################################################################################
# ------------------------------ Helper functions ------------------------------
################################################################################


@njit(parallel=True)
def _t_statistic_vectorized_numba(
        noise_samples: np.ndarray,
        planet_samples: np.ndarray
) -> np.ndarray:
    """
    Fast and parallel version of t_statistic_vectorized using numba. More
    information given in t_statistic_vectorized.
    """
    noise_size_n = noise_samples.shape[1]
    planet_size_m = 1

    res = []
    for i in range(noise_samples.shape[0]):
        x_bar_noise = noise_samples[i, :].mean()
        y_bar_planet = planet_samples[i]

        # np.sqrt(float(n) / float(n-1)) is the ddof = 1
        noise_std = noise_samples[i, :].std() * np.sqrt(
            1 / noise_size_n + 1 / planet_size_m) * \
            np.sqrt(float(noise_size_n) / float(noise_size_n - 1))
        statistic_t = (y_bar_planet - x_bar_noise) / noise_std
        res.append(statistic_t)

    return np.array(res)


def t_statistic_vectorized(
        noise_samples: Union[List[float], np.ndarray],
        planet_samples: Union[float, np.ndarray],
        numba_parallel_threads: int = 1
) -> Union[float, np.ndarray]:
    """
    Computes the test-statistic of the ttest / bootstrapping using vectorized
    code. Usually noise_samples and planet_samples contain a list of
    multiple samples. For every pair of noise and planet sample this function
    returns one T value. In case more than 10e4 test values have to be computed
    a parallel implementation using numba can be used.

    Args:
        planet_samples: The planet sample containing the observation of the
            planet :math:`Y_1,` as a float.
            In case multiple tests are performed the input can also be a
            list or 1D array :math:`((Y_1)_1, (Y_1)_2, ...)`
        noise_samples: The noise observations containing
            :math:`(X_1, ..., X_n)`. In case multiple tests are
            performed the input can also be a list of lists / 1D arrays or
            a 2D array:
            :math:`(X_1, ..., X_n)_1, (X_1, ..., X_n)_2, ...`.
        numba_parallel_threads: The number of parallel threads used by numba.
            In case the function is used with multiprocessing this number
            should always be equal to the default of 1.

    Returns:
        The ttest / bootstrapping test statistic :math:`T_{obs},`. Either a
        single value or a 1D numpy array.
    """

    # make sure noise_samples is np.array
    noise_samples = np.array(noise_samples)

    if isinstance(planet_samples, (np.floating, float)):
        # Only a single test
        planet_size_m = 1
        noise_size_n = noise_samples.shape[0]
        x_bar_noise = np.mean(noise_samples)
        y_bar_planet = planet_samples
        noise = np.std(noise_samples, ddof=1) * \
            np.sqrt(1 / noise_size_n + 1 / planet_size_m)

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

        planet_size_m = 1
        noise_size_n = noise_samples.shape[1]
        x_bar_noise = np.mean(noise_samples, axis=1)
        y_bar_planet = planet_samples
        noise = np.std(noise_samples, axis=1, ddof=1) *\
            np.sqrt(1 / noise_size_n + 1 / planet_size_m)

    statistic_t = (y_bar_planet - x_bar_noise) / noise
    return statistic_t

################################################################################
# ------------------------------ The Test --------------------------------------
################################################################################


class TTest(TestInterface):
    """
    A classical two sample ttest. Assumes iid Gaussian noise and tests for
    differences in means.
    """

    @classmethod
    def fpf_2_t(
            cls,
            fpf: Union[float, np.ndarray],
            num_noise_values: int
    ) -> Union[float, np.ndarray]:
        """
        Computes the required value of :math:`T_{obs},` (the test statistic) to
        get a confidence level of fpf. Takes into account the effect of the
        sample size by using the t-distribution.
        Accepts a single value as input as well as a list of fpf values.

        Args:
            fpf: Desired confidence level(s) as FPF
            num_noise_values: Number of noise observations. Needed to take the
                effect of the sample size into account.

        Returns:
            The required value(s) of :math:`T_{obs},`
        """
        noise_size_n = num_noise_values
        planet_size_m = 1

        statistic_t = stats.t.isf(fpf, df=noise_size_n + planet_size_m - 2)
        return statistic_t

    def t_2_fpf(
            self,
            statistic_t: Union[float, np.ndarray],
            num_noise_values: int
    ) -> Union[float, np.ndarray]:
        """
        Computes the p-value of the ttest given the test statistic
        :math:`T_{obs},`. Takes into account the effect of the sample size by
        using the t-distribution.
        Accepts a single value as input as well as a list of :math:`T_{obs},`
        values.

        Args:
            statistic_t: The test statistic value(s) :math:`T_{obs},`
            num_noise_values: Number of noise observations. Needed to take the
                effect of the sample size into account.

        Returns:
            The uncertainty / p-value / fpf of the test

        """
        noise_size_n = num_noise_values
        planet_size_m = 1

        if isinstance(statistic_t, (float, np.floating)):
            fpf = stats.t.sf(statistic_t, df=noise_size_n + planet_size_m - 2)

        # check if we can use multiprocessing for speedups
        elif len(statistic_t) > 10e4:
            # split the t values into 100 sub arrays and run them in parallel

            with multiprocessing.Pool(int(self.num_cpus)) as pool:
                mp_results = pool.starmap(
                    stats.t.sf,
                    [(sub_array, planet_size_m + noise_size_n - 2)
                     for sub_array in
                     np.array_split(statistic_t, 100)])

            fpf = np.concatenate(mp_results).flatten()
        else:
            fpf = stats.t.sf(statistic_t, df=noise_size_n + planet_size_m - 2)

        return fpf

    def test_2samp(
            self,
            planet_samples: Union[float, np.ndarray],
            noise_samples: Union[List[float], np.ndarray]
    ) -> Union[Tuple[float, float],
               Tuple[np.ndarray, np.ndarray]]:
        r"""
        Performs one (or multiple) two sample T-Tests given noise observations
        :math:`(X_1, ..., X_n)` and a planet observation :math:`Y_1,`
        with null hypothesis:

        :math:`H_0: \mu_X = \mu_Y`

        against the alternative:

        :math:`H_1: \mu_X < \mu_Y`

        This implementation is similar to the one in scipy but calculates the
        special case where only one value is given in the planet sample.

        Args:
            planet_samples: The planet sample containing the observation of the
                planet :math:`Y_1` as a float.
                In case multiple tests are performed the input can also be a
                list or 1D array :math:`((Y_1)_1, (Y_1)_2, ...)`.
            noise_samples: The noise observations containing
                :math:`(X_1, ..., X_n)`. In case multiple tests are
                performed the input can also be a list of lists / 1D arrays or
                a 2D array:
                :math:`(X_1, ..., X_n)_1, (X_1, ..., X_n)_2, ...`

        Returns:
            1. The p-value (or a 1d array of p-values) of the test i.e. the FPF.

            2. The test statistic :math:`T_{obs},` (or a 1d array of
            the statistics).
        """

        # run the super method to check if the shapes are correct
        super().test_2samp(planet_samples, noise_samples)

        # make sure noise_samples is np.array
        noise_samples = np.array(noise_samples)

        statistic_t = t_statistic_vectorized(
            noise_samples,
            planet_samples,
            numba_parallel_threads=self.num_cpus)

        if isinstance(planet_samples, (np.floating, float)):
            num_noise_values = noise_samples.shape[0]
        else:
            num_noise_values = noise_samples.shape[1]

        p_values = self.t_2_fpf(statistic_t, num_noise_values)

        return p_values, statistic_t

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
        :math:`X_n` such that a t-test with noise
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
            desired_confidence_fpf :math:`f` for a t-test.
            In case of multiple test the output is a 1D array.
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
        for i, tmp_noise_at_planet_pos in enumerate(noise_at_planet_pos):
            tmp_noise_samples = noise_samples[i]

            planet_size_m = 1
            noise_size_n = len(tmp_noise_samples)

            tmp_t = self.fpf_2_t(desired_confidence_fpf,
                                 noise_size_n)
            tmp_sigma = np.std(tmp_noise_samples, ddof=1)
            tmp_noise = tmp_sigma * np.sqrt(1 / noise_size_n + 1 /
                                            planet_size_m)
            tmp_x_bar = np.mean(tmp_noise_samples)

            residual_flux_needed = tmp_t * tmp_noise + tmp_x_bar
            residual_flux_needed -= tmp_noise_at_planet_pos
            all_flux_needed.append(residual_flux_needed)

        if len(all_flux_needed) == 1:
            return all_flux_needed[0]

        return np.array(all_flux_needed)
