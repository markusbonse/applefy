"""
This module contains implementations of parametric bootstrapping tests which can
be used for to calculate contrast curves and contrast grids.
"""

from typing import Union, List, Optional

import json
import multiprocessing
from abc import abstractmethod

import numpy as np
from scipy import interpolate

try:
    from parallel_sort import parallel_sort_inplace as sort_fast
    FOUND_FAST_SORT = True
except ImportError:
    sort_fast = np.sort
    FOUND_FAST_SORT = False

from applefy.statistics.general import gaussian_sigma_2_fpf
from applefy.statistics.parametric import TTest, t_statistic_vectorized

################################################################################
# ----------------- General interface for Bootstrap Tests ----------------------
################################################################################


class BootstrapTest(TTest):
    """
    This is a general interface for all bootstrap tests. It implements all
    functionality needed to save and load bootstrap results as json files and
    to run a new bootstrap experiment. It extends the classical
    :meth:`~applefy.statistics.parametric.TTest` as both use the same test
    statistic :math:`T_{obs}`. But for a given value of :math:`T_{obs}` the
    bootstrap tests will give a different p-value / fpf.
    """

    def __init__(
            self,
            noise_observations: Union[List[float], np.ndarray, None],
            num_cpus: int = 1):
        """
        Constructor of a BootstrapTest.

        Args:
            noise_observations: Noise observations which are used to run the
                bootstrap experiments. This can either be a single sample of
                noise observations :math:`(X_1, ..., X_n)` or a numpy array
                of several samples
                :math:`(X_1, ..., X_n)_1, (X_1, ..., X_n)_2, ...`.
                It can also be set to None. If set to None
                :meth:`~run_bootstrap_experiment` can not be used which is why
                this option should only be used to restore a previously run
                bootstrap experiment from a .json file
                (see :meth:`~restore_lookups`).
            num_cpus: The number of CPU cores that will be used in all tests
                (e.g. to run the bootstrapping).
        """

        super().__init__(num_cpus)

        if not isinstance(noise_observations, (list, np.ndarray)):

            if noise_observations is not None:
                raise ValueError("noise_observations have to be either a list /"
                                 " 1D np.array of noise observations or a list"
                                 " of lists / 1D np.arrays")

        self.lookup_tables = dict()
        self.noise_observations = noise_observations
        if self.noise_observations is None:
            return

        # If we are in the special case of a single noise observation we have
        # to unify the inputs
        if isinstance(noise_observations, np.ndarray):
            if len(noise_observations.shape) == 1:
                self.noise_observations = [noise_observations, ]

        if isinstance(noise_observations, list):
            if not isinstance(noise_observations[0], (list, np.ndarray)):
                self.noise_observations = [np.array(noise_observations), ]

    @classmethod
    def construct_from_json_file(
            cls,
            lookup_file: str
    ) -> "BootstrapTest":
        """
        An alternative constructor to create a BootstrapTest from a previously
        calculated bootstrap experiment given as a .json file. The function
        will restore the lookup table which maps :math:`T_{obs}` to the p-values
        and vice versa.

        Args:
            lookup_file: A path to a .json file containing the statistics to be
                restored (lookup tables).

        Returns:
            Instance of the BootstrapTest with restored lookup table.

        """

        bootstrap_test = cls(None)
        bootstrap_test.restore_lookups(lookup_file)
        return bootstrap_test

    # functions for loading and saving lookup tables
    def restore_lookups(
            self,
            lookup_file: str
    ) -> None:
        """
        Restores previously computed bootstrap results / lookup tables from a
        .json file. The lookup table maps :math:`T_{obs}` to the p-values and
        vice versa.
        If lookups already exist they are updated. Duplicates are overwritten.

        Args:
            lookup_file: A path to a .json file containing the lookup tables.

        """

        # load the file
        with open(lookup_file) as file:
            json_lookups = json.load(file)

        # convert t and fpf lists to np.arrays
        lookups_new = dict()
        for key, values in json_lookups.items():
            tmp_dict = {"t": np.array(values["t"]),
                        "fpf": np.array(values["fpf"])}
            lookups_new[int(key)] = tmp_dict

        self.lookup_tables.update(lookups_new)

    def save_lookups(
            self,
            lookup_file: str
    ) -> None:
        """
        Saves the internal lookup tables into a .json file.
        The lookup tables map :math:`T_{obs}` to the p-values and vice versa.
        The saved tables can be restored with :meth:`~restore_lookups`.

        Args:
            lookup_file: The path with filename where lookup tables are saved.
        """

        # internally the values of t and fpf are np.arrays.
        # We have to convert them into lists
        json_lookups = dict()
        for key, values in self.lookup_tables.items():
            tmp_dict = {"t": list(values["t"]),
                        "fpf": list(values["fpf"])}
            json_lookups[key] = tmp_dict

        # Save the results
        with open(lookup_file, 'w') as file:
            json.dump(json_lookups, file)

    # functions to run bootstrapping
    def run_bootstrap_experiment(
            self,
            memory_size: int,
            num_noise_values: int,
            num_draws: int = 10e8,
            approximation_interval: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""
        Runs a bootstrapping experiment (resampling) in order to calculate
        the distribution of the test statistic :math:`T` under
        :math:`H_0` given a sample size of m=1 (one planet observation) and
        n=num_noise_values. Allows the use of multiprocessing and management of
        memory size. The result is stored as a lookup table using the
        approximation_interval.
        The strategy used to resample during the bootstrapping is implemented
        by the classes inheriting from this class.

        Args:
            memory_size: Maximum number of float values stored per process.
                A loop is used in case the number is small.
            num_noise_values: The sample size of the noise observations. This
                depends on the separation from the star.
            num_draws: Number of bootstrap experiments (resamples) :math:`B`.
            approximation_interval: The values in terms of
                :math:`\sigma_{\mathcal{N}}` at which the distribution of
                :math:`T_{obs}` is evaluated and stored as a lookup table. If
                None a np.linspace(-7, 7, 10000) will be used.

        Returns:
            A 1D array with :math:`B` bootstrap values :math:`T^*`.

        """
        # set the approximation_interval if it is None
        if approximation_interval is None:
            approximation_interval = np.linspace(-7, 7, 10000)

        # make sure num_noise_values is an int. Otherwise, saving as json might
        # fail later
        num_noise_values = int(num_noise_values)

        if self.noise_observations is None:
            raise ValueError("Can not run bootstrap experiment without data"
                             " to bootstrap from.")

        num_experiments = int(num_draws / memory_size)

        with multiprocessing.Pool(int(self.num_cpus)) as pool:
            mp_results = pool.starmap(
                self._run_boostrap_mp,
                [(int(memory_size), int(num_noise_values)), ] * num_experiments)

        t_results = np.concatenate(mp_results)

        # We have to sort the t values in order to allow for fast computation
        # of fpf and confidence levels later
        if FOUND_FAST_SORT:
            sort_fast(t_results)
        else:
            t_results.sort()

        # The distribution of t under H0 is directly given by the t_results
        # we have computed. However, these values take a lot of memory. Thus,
        # we approximate the distribution of t by evaluating paris of t and
        # fpf for different fpf
        fpf_approx = gaussian_sigma_2_fpf(approximation_interval)

        # Approximate t by lookup in the t_results

        # Slow version on non-sorted self.t_results
        # needed_planet_flux = np.quantile(self.t_results, 1 - fpf)
        # Note: We use liner interpolation to approximate the quantiles

        # compute the idx where to look up the t values
        tmp_idx = (len(t_results) - 1) * (1 - fpf_approx)
        floor_value = np.floor(tmp_idx)
        ceil_value = np.ceil(tmp_idx)

        t_floor = t_results[floor_value.astype(np.int64)] * \
            (ceil_value - tmp_idx)
        t_ceil = t_results[ceil_value.astype(np.int64)] * \
            (tmp_idx - floor_value)

        self.lookup_tables[num_noise_values] = dict()
        self.lookup_tables[num_noise_values]["t"] = t_floor + t_ceil
        self.lookup_tables[num_noise_values]["fpf"] = fpf_approx
        return t_results

    def _run_boostrap_mp(
            self,
            num_draws: int,
            num_noise_values: int
    ) -> np.ndarray:
        """
        Internal function used to resample and calculate the test statistic
        :math:`T` with multiprocessing. This function is executed in
        parallel in :meth:`~run_bootstrap_experiment`.

        Args:
            num_draws: Number of resamples :math:`B`.
            num_noise_values: The sample size of the noise observations. This
                depends on the separation from the star.

        Returns:
            1D array with :math:`B` values of  :math:`T^*`.

        """
        np.random.seed()

        # We might have different noise observations we can use to sample values
        # of the test statistic. This code randomly samples from the different
        # observation lists and bootstraps from them. The actual distribution
        # we sample from is defined by the child classes (e.g. Gaussian, Laplace
        # with replacement from the observations ...)

        # We draw num_draws indices to select which lists of observations we use
        # for each computation of t.
        idx_list = np.random.randint(0,
                                     len(self.noise_observations),
                                     num_draws)
        # Example sub_idx[0] = 3 and sub_num_draws[0] = 10 mean that we sample
        # 10 times using the observation list / array at idx 3 in the
        # local_observation_list
        sub_idx, sub_num_draws = np.unique(idx_list, return_counts=True)

        t_values = []

        for i, tmp_observation_list_idx in enumerate(sub_idx):
            tmp_num_draws = sub_num_draws[i]

            tmp_t = self._sample_t(tmp_observation_list_idx,
                                   tmp_num_draws,
                                   num_noise_values)

            t_values.append(tmp_t)

        return np.concatenate(t_values).flatten()

    @abstractmethod
    def _sample_t(
            self,
            observation_list_idx: int,
            num_draws: int,
            num_noise_values: int
    ) -> np.ndarray:
        """
        Abstract interface which is implemented by the inheriting classes. This
        function contains how we resample during bootstrapping and is
        different depending on the type of noise we assume.

        Args:
            observation_list_idx: Index of which of the available input noise
                observations is used to resample. This is only used if a list
                of (list, 1D array) was passed as noise observations during
                initialization of the test.
            num_draws: Number of resamples :math:`B`.
            num_noise_values: The sample size of the noise observations. This
                depends on the separation from the star.

        Returns:
            1D array with :math:`B` values of  :math:`T^*`.

        """
        return np.array([0,])

    # functions to compute the tests
    def t_2_fpf(
            self,
            statistic_t: Union[float, np.ndarray],
            num_noise_values: int
    ) -> Union[float, np.ndarray]:
        """
        Computes the p-value of the ttest given the test statistic
        :math:`T_{obs},`. Takes into account the effect of the sample size and
        type of the noise (using previously computed lookup tables).
        Accepts a single value as input as well as a list of :math:`T_{obs},`
        values.

        Args:
            statistic_t: The test statistic value(s) :math:`T_{obs},`
            num_noise_values: Number of noise observations. Needed to take the
                effect of the sample size into account.

        Returns:
            The uncertainty / p-value / fpf of the test

        """

        if num_noise_values not in self.lookup_tables.keys():
            raise ValueError("No bootstrapping distribution of t available "
                             "for " + str(num_noise_values) + " noise values."
                             " Please run a new bootstrap experiment or restore"
                             " results from CSV files.")

        t_lookup = self.lookup_tables[num_noise_values]["t"]
        fpf_lookup = self.lookup_tables[num_noise_values]["fpf"]

        # interpolate the results from the lookup table
        t2fpf = interpolate.interp1d(t_lookup,
                                       fpf_lookup,
                                       kind="cubic",
                                       fill_value="extrapolate")

        if isinstance(statistic_t, (float, np.floating)):
            fpf = t2fpf(statistic_t)

        # check if we can use multiprocessing for speedups
        elif len(statistic_t) > 10e4:
            # split the t values into 100 sub arrays and run them in parallel

            with multiprocessing.Pool(int(self.num_cpus)) as pool:
                mp_results = pool.starmap(
                    t2fpf,
                    [(sub_array, ) for sub_array in
                     np.array_split(statistic_t, 100)])

            fpf = np.concatenate(mp_results).flatten()
        else:
            fpf = t2fpf(statistic_t)

        return fpf

    def fpf_2_t(
            self,
            fpf: Union[float, np.ndarray],
            num_noise_values: int
    ) -> Union[float, np.ndarray]:
        """
        Computes the required value of :math:`T_{obs},` (the test statistic) to
        get a confidence level of fpf.
        Takes into account the effect of the sample size and
        type of the noise (using previously computed lookup tables).
        Accepts a single value as input as well as a list of fpf values.

        Args:
            fpf: Desired confidence level(s) as FPF
            num_noise_values: Number of noise observations. Needed to take the
                effect of the sample size into account.

        Returns:
            The required value(s) of :math:`T_{obs},`
        """

        if num_noise_values not in self.lookup_tables.keys():
            raise ValueError("No bootstrapping distribution of t available "
                             "for " + str(num_noise_values) + " noise values."
                             " Please run a new bootstrap experiment or restore"
                             " results from CSV files.")

        t_lookup = self.lookup_tables[num_noise_values]["t"]
        fpf_lookup = self.lookup_tables[num_noise_values]["fpf"]

        # interpolate the results from the lookup table
        fpf2t = interpolate.interp1d(
            fpf_lookup,
            t_lookup,
            kind="cubic",
            fill_value="extrapolate")

        return fpf2t(fpf)

################################################################################
# -----------------Parametric Bootstrap Test -----------------------------------
################################################################################


class GaussianBootstrapTest(BootstrapTest):
    """
    The GaussianBootstrapTest is a parametric hypothesis test which assumes that
    the noise is Gaussian. This test is approximately equivalent to the ttest.
    This implementation is only for illustration purposes and should not be
    used in practice.
    """

    def _sample_t(
            self,
            observation_list_idx: int,
            num_draws: int,
            num_noise_values: int
    ) -> np.ndarray:
        """
        Uses Parametric bootstrapping to resample from a Gaussian distribution.

        Args:
            observation_list_idx: Index of which of the available input noise
                observations is used to resample. This is only used if a list
                of (list, 1D array) was passed as noise observations during
                initialization of the test.
            num_draws: Number of resamples :math:`B`.
            num_noise_values: The sample size of the noise observations. This
                depends on the separation from the star.

        Returns:
            1D array with :math:`B` values of  :math:`T^*`.

        """

        noise_values = self.noise_observations[observation_list_idx]

        # Step 1: Compute the MLE parameters of the noise distribution under H0
        # Note: As shown in the paper the test statistic t under H0 is
        # independent of the location and scale of the noise distribution.
        # We only calculate it here for illustration purposes as the property
        # does not hold for non-scale shift family distributions.
        location_mu = np.mean(noise_values)
        scale_sigma = np.std(noise_values, ddof=1)

        # Step 2: Sample Noise
        # here we could use the standard gaussian as well with location = 0
        # and scale = 1.
        x_boot = np.random.normal(loc=location_mu,
                                  scale=scale_sigma,
                                  size=(num_draws, num_noise_values))

        # Step 3: Sample Planets
        y_boot = np.random.normal(loc=location_mu,
                                  scale=scale_sigma,
                                  size=num_draws)

        return t_statistic_vectorized(x_boot, y_boot)


class LaplaceBootstrapTest(BootstrapTest):
    """
    The LaplaceBootstrapTest is a parametric hypothesis test which assumes that
    the distribution of the noise is Laplace. The test accounts for the higher
    occurrence rate of bright noise values as well as for the small sample size
    at close separations to the star. Applefy comes with
    `previously computed <../02_user_documentation/03_bootstrapping.ipynb>`_
    lookup tables.
    """

    def __init__(
            self,
            noise_observations: Union[List[float], np.ndarray, None] = None,
            num_cpus: int = 1):
        """
        Constructor of a LaplaceBootstrapTest.

        Args:
            noise_observations: Noise observations which are used to run the
                bootstrap experiments. The LaplaceBootstrapTest benefits from
                pivoting. Hence, noise_observations will have no effect on the
                result.

            num_cpus: The number of CPU cores that will be used in all tests
                (e.g. to run the bootstrapping).
        """

        if noise_observations is None:
            noise_observations = np.random.laplace(0, 1, 6)

        super().__init__(noise_observations, num_cpus)


    def _sample_t(
            self,
            observation_list_idx,
            num_draws,
            num_noise_values):
        """
        Uses Parametric bootstrapping to resample from a Laplacian distribution
        and calculate the values of :math:`T^*`.

        Args:
            observation_list_idx: Index of which of the available input noise
                observations is used to resample. This is only used if a list
                of (list, 1D array) was passed as noise observations during
                initialization of the test.
            num_draws: Number of resamples :math:`B`.
            num_noise_values: The sample size of the noise observations. This
                depends on the separation from the star.

        Returns:
            1D array with :math:`B` values of  :math:`T^*`.

        """

        noise_values = self.noise_observations[observation_list_idx]

        # Step 1: Compute the MLE parameters of the noise distribution under H0
        # Note: As shown in the paper the test statistic t under H0 is
        # independent of the location and scale of the noise distribution.
        # We only calculate it here for illustration purposes as the property
        # does not hold for non-scale shift family distributions.
        location_mu = np.median(noise_values)
        scale_b = np.mean(np.abs(noise_values - location_mu))

        # Step 2: Sample Noise
        # here we could use the standard laplace as well with location = 0
        # and scale = 1.
        x_boot = np.random.laplace(loc=location_mu,
                                   scale=scale_b,
                                   size=(num_draws, num_noise_values))

        # Step 3: Sample Planets
        y_boot = np.random.laplace(loc=location_mu,
                                   scale=scale_b,
                                   size=num_draws)

        return t_statistic_vectorized(x_boot, y_boot)
