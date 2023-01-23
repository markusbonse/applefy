"""
Parametric and non-parametric bootstrapping tests.
"""

import json
import numpy as np
from scipy import interpolate
import multiprocessing
from abc import abstractmethod

try:
    from parallel_sort import parallel_sort_inplace as sort_fast
    found_fast_sort = True
except ImportError:
    sort_fast = np.sort
    found_fast_sort = False

from applefy.statistics.general import gaussian_sigma_2_fpf
from applefy.statistics.parametric import TTest, t_statistic_vectorized

################################################################################
# ----------------- General interface for Bootstrap Tests ----------------------
################################################################################


class BootstrapTest(TTest):
    """
    This is a general interface for all bootstrap tests. It implements all
    functionality needed to save and load bootstrap results as json files and
    run a new bootstrapping experiment. It extends the classical TTest class
    as both use the same test statistic. They only differ in the mapping from
    fpf to statistic and vice versa.
    """

    def __init__(self,
                 noise_observations,
                 num_cpus=1):
        """
        Constructor of a BootstrapTest.

        Args:
            noise_observations: Noise observations which are used to run the
                bootstrap experiments. This can either be a single sample of
                noise observations or a list of samples. (list, 1D array) or
                list of (list, 1D array). Can also be set to None. This prevents
                to run new bootstrap experiments. Only makes sense if previously
                evaluated statistics are restored from json files.
            num_cpus: The number of CPU core that will be used during all
                experiments.

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
    def construct_from_json_file(cls,
                                 lookup_file):
        """
        An alternative constructor to create a BootstrapTest based on previously
        calculated bootstrap results given as a .json file.

        Args:
            lookup_file: A path to a .json file containing the statistics to be
                restored (lookup tables).

        Returns: Instance of BootstrapTest

        """

        bootstrap_test = cls(None)
        bootstrap_test.restore_lookups(lookup_file)
        return bootstrap_test

    # functions for loading and saving lookup tables
    def restore_lookups(self,
                        lookup_file):
        """
        Restores previous bootstrap results / lookup tables from a .json file.
        The set of already available lookup tables is updated. Duplicated are
        overwritten.

        Args:
            lookup_file: A path to a .json file containing the statistics to be
                restored (lookup tables).

        """

        # load the file
        with open(lookup_file) as f:
            json_lookups = json.load(f)

        # convert tau and fpf lists to np.arrays
        lookups_new = dict()
        for key, values in json_lookups.items():
            tmp_dict = {"tau": np.array(values["tau"]),
                        "fpf": np.array(values["fpf"])}
            lookups_new[int(key)] = tmp_dict

        self.lookup_tables.update(lookups_new)

    def save_lookups(self,
                     lookup_file):
        """
        Saves the internal lookup tables into a .json file. This way they can be
        restored using restore_lookups on a later stage.

        Args:
            lookup_file: The path / filename where to store the lookup tables.

        """

        # internally the values of tau and fpf are np.arrays.
        # We have to convert them into lists
        json_lookups = dict()
        for key, values in self.lookup_tables.items():
            tmp_dict = {"tau": list(values["tau"]),
                        "fpf": list(values["fpf"])}
            json_lookups[key] = tmp_dict

        # Save the results
        with open(lookup_file, 'w') as f:
            json.dump(json_lookups, f)

    # functions to run bootstrapping
    def run_bootstrap_experiment(self,
                                 memory_size,
                                 num_noise_values,
                                 num_draws=10e8,
                                 approximation_interval=
                                 np.linspace(-7, 7, 10000)):
        """
        Calculates the distribution of the test statistic tau under H0 for a
        sample size of m=1 (one planet) and n=num_noise_values by using
        bootstrapping. Allows the use of multiprocessing and management of the
        memory size. The result is approximated and a lookup table is stored.
        The strategy used to resample during the bootstrapping is implemented
        by the classes inheriting from this class.

        Args:
            memory_size: Maximum number of float values stored per process.
                A loop is used in case the number is small.
            num_noise_values: The sample size of the noise.
            num_draws: Number of bootstrap experiments B
            approximation_interval: The values in terms of gaussian sigma at
                which the results are approximated and stored in the internal
                lookup table.

        Returns: A 1D array containing all B tau-values generated during
            bootstrapping

        """

        # make sure num_noise_values is an int. Otherwise saving as json might
        # fail later
        num_noise_values = int(num_noise_values)

        if self.noise_observations is None:
            raise ValueError("Can not run bootstrap experiment without data"
                             " to bootstrap from.")

        num_experiments = int(num_draws / memory_size)

        pool = multiprocessing.Pool(int(self.num_cpus))
        mp_results = pool.starmap(
            self._run_boostrap_mp,
            [(int(memory_size), int(num_noise_values)), ] * num_experiments)

        pool.close()
        tau_results = np.concatenate(mp_results)

        # We have to sort the t values in order to allow for fast computation
        # of fpf and confidence levels later
        if found_fast_sort:
            sort_fast(tau_results)
        else:
            tau_results.sort()

        # The distribution of tau under H0 is directly given by the tau_results
        # we have computed. However, these values take a lot of memory. Thus,
        # we approximate the distribution of tau by evaluating paris of tau and
        # fpf for different fpf
        fpf_approx = gaussian_sigma_2_fpf(approximation_interval)

        # Approximate tau by lookup in the tau_results

        # Slow version on non-sorted self.t_results
        # needed_planet_flux = np.quantile(self.t_results, 1 - fpf)
        # Note: We use liner interpolation to approximate the quantiles

        # compute the idx where to look up the tau values
        k = (len(tau_results) - 1) * (1 - fpf_approx)
        f = np.floor(k)
        c = np.ceil(k)

        d0 = tau_results[f.astype(np.int64)] * (c - k)
        d1 = tau_results[c.astype(np.int64)] * (k - f)

        tau_approx = d1 + d0

        self.lookup_tables[num_noise_values] = dict()
        self.lookup_tables[num_noise_values]["tau"] = tau_approx
        self.lookup_tables[num_noise_values]["fpf"] = fpf_approx
        return tau_results

    def _run_boostrap_mp(self,
                         num_draws,
                         num_noise_values):
        """
        Internal function used to resample and calculate the test statistic tau.
        This function is executed in parallel in run_bootstrap_experiment.

        Args:
            num_draws: Number of tau values to be calculated.
            num_noise_values: Number of noise observations i.e. sample size.

        Returns: 1D array of tau values.

        """
        np.random.seed()

        # We might have different noise observations we can use to sample values
        # of the test statistic. This code randomly samples from the different
        # observation lists and bootstraps from them. The actual distribution
        # we sample from is defined by the child classes (e.g. Gaussian, Laplace
        # with replacement from the observations ...)

        # We draw num_draws indices to select which lists of observations we use
        # for each computation of tau.
        idx_list = np.random.randint(0,
                                     len(self.noise_observations),
                                     num_draws)
        # Example sub_idx[0] = 3 and sub_num_draws[0] = 10 mean that we sample
        # 10 times using the observation list / array at idx 3 in the
        # local_observation_list
        sub_idx, sub_num_draws = np.unique(idx_list, return_counts=True)

        tau_values = []

        for i in range(len(sub_idx)):
            tmp_observation_list_idx = sub_idx[i]
            tmp_num_draws = sub_num_draws[i]

            tmp_tau = self._sample_tau(tmp_observation_list_idx,
                                       tmp_num_draws,
                                       num_noise_values)

            tau_values.append(tmp_tau)

        return np.concatenate(tau_values).flatten()

    @abstractmethod
    def _sample_tau(self,
                    observation_list_idx,
                    num_draws,
                    num_noise_values):
        """
        Abstract interface which is implemented by the inheriting classes. This
        function contains how we resample during the bootstrapping and is
        different for parametric, semi-parametric and non-parametric
        bootstrapping.

        Args:
            observation_list_idx: Index of which of the available input noise
                observations is used to resample. This is only used if a list
                of (list, 1D array) if was passed as noise observations during
                initialization of the test.
            num_draws: Number of tau values to be calculated.
            num_noise_values: Number of noise observations i.e. sample size.

        Returns: resampled and evaluated values of tau

        """
        return 0

    # functions to compute the tests
    def tau2fpf(self,
                tau,
                num_noise_values):
        """
        Computed the confidence as fpf given the test statistic tau. Takes into
        account the effect of the sample size and type of the noise by using
        the previously computed lookup tables.
        Accepts single value inputs as well as a list of fpf values.

        Args:
            tau: The test statistic (float or list)
            num_noise_values: Number of noise observations. (int)

        Returns: The confidence / p-value / fpf of the test

        """

        if num_noise_values not in self.lookup_tables.keys():
            raise ValueError("No bootstrapping distribution of tau available "
                             "for " + str(num_noise_values) + " noise values."
                             " Please run a new bootstrap experiment or restore"
                             " results from CSV files.")

        tau_lookup = self.lookup_tables[num_noise_values]["tau"]
        fpf_lookup = self.lookup_tables[num_noise_values]["fpf"]

        # interpolate the results from the lookup table
        tau2fpf = interpolate.interp1d(tau_lookup,
                                       fpf_lookup,
                                       kind="cubic",
                                       fill_value="extrapolate")

        if isinstance(tau, (float, np.floating)):
            fpf = tau2fpf(tau)

        # check if we can use multiprocessing for speedups
        elif len(tau) > 10e4:
            # split the tau values into 100 sub arrays and run them in parallel
            pool = multiprocessing.Pool(int(self.num_cpus))
            mp_results = pool.starmap(
                tau2fpf,
                [(sub_array, ) for sub_array in
                    np.array_split(tau, 100)])

            pool.close()

            fpf = np.concatenate(mp_results).flatten()
        else:
            fpf = tau2fpf(tau)

        return fpf

    def fpf2tau(self,
                fpf,
                num_noise_values):
        """
        Computes the required value of tau (the test statistic) to get a
        confidence level of fpf. Takes into account the effect of the sample
        size and type of the noise by using the previously computed lookup
        tables.
        Accepts single value inputs as well as a list of fpf values.

        Args:
            fpf: Desired confidence level as FPF (float or list)
            num_noise_values: Number of noise observations. (int)

        Returns: The needed test statistic tau (float or list of floats)
        """

        if num_noise_values not in self.lookup_tables.keys():
            raise ValueError("No bootstrapping distribution of tau available "
                             "for " + str(num_noise_values) + " noise values."
                             " Please run a new bootstrap experiment or restore"
                             " results from CSV files.")

        tau_lookup = self.lookup_tables[num_noise_values]["tau"]
        fpf_lookup = self.lookup_tables[num_noise_values]["fpf"]

        # interpolate the results from the lookup table
        fpf2tau = interpolate.interp1d(fpf_lookup,
                                       tau_lookup,
                                       kind="cubic",
                                       fill_value="extrapolate")

        return fpf2tau(fpf)

################################################################################
# -----------------Parametric Bootstrap Test -----------------------------------
################################################################################


class GaussianBootstrapTest(BootstrapTest):
    """
    The GaussianBootstrapTest is a parametric hypothesis test which assumes that
    the distribution of the noise is Gaussian. This test is approximately
    equivalent to the ttest. This implementation is only for illustration
    purposes and should not be used in practice.
    """

    def _sample_tau(self,
                    observation_list_idx,
                    num_draws,
                    num_noise_values):
        """
        Uses Parametric bootstrapping to resample from a Gaussian distribution.

        Args:
            observation_list_idx: Index of which of the available input noise
                observations is used to resample. This is only used if a list
                of (list, 1D array) if was passed as noise observations during
                initialization of the test.
            num_draws: Number of tau values to be calculated.
            num_noise_values: Number of noise observations i.e. sample size.

        Returns: resampled and evaluated values of tau

        """

        noise_values = self.noise_observations[observation_list_idx]

        # Step 1: Compute the MLE parameters of the noise distribution under H0
        # Note: As shown in the paper the test statistic tau under H0 is
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
    at close separations to the star.
    """

    def _sample_tau(self,
                    observation_list_idx,
                    num_draws,
                    num_noise_values):
        """
        Uses Parametric bootstrapping to resample from a Laplacian distribution.

        Args:
            observation_list_idx: Index of which of the available input noise
                observations is used to resample. This is only used if a list
                of (list, 1D array) if was passed as noise observations during
                initialization of the test.
            num_draws: Number of tau values to be calculated.
            num_noise_values: Number of noise observations i.e. sample size.

        Returns: resampled and evaluated values of tau

        """

        noise_values = self.noise_observations[observation_list_idx]

        # Step 1: Compute the MLE parameters of the noise distribution under H0
        # Note: As shown in the paper the test statistic tau under H0 is
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
