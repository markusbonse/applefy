"""
The contrast module is the main interface to compute contrast curves with
applefy. For an example on how to use this module see the
`user documentation <../02_user_documentation/01_contrast_curves.ipynb>`_
"""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Tuple, Any

from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from applefy.utils.file_handling import save_as_fits, open_fits, \
    create_checkpoint_folders, search_for_config_and_residual_files
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics.general import TestInterface

from applefy.utils.file_handling import save_experiment_configs, \
    read_fake_planet_results
from applefy.utils.fake_planets import calculate_fake_planet_positions, \
    generate_fake_planet_experiments, add_fake_planets
from applefy.utils.photometry import estimate_stellar_flux
from applefy.utils.fake_planets import sort_fake_planet_results, \
    merge_fake_planet_residuals
from applefy.utils.throughput import compute_throughput_table
from applefy.utils.contrast_curve import compute_contrast_curve
from applefy.utils.contrast_grid import compute_contrast_grid, \
    compute_contrast_from_grid


class Contrast:
    """
    The Contrast class is the main interface to compute contrast curves with
    applefy. It allows to compute contrast grids as well as classical
    analytical contrast curves. For an example on how to use this class see the
    `user documentation <../02_user_documentation/01_contrast_curves.ipynb>`_
    """

    def __init__(
            self,
            science_sequence: np.ndarray,
            psf_template: np.ndarray,
            psf_fwhm_radius: float,
            parang_rad: np.ndarray,
            dit_science: float,
            dit_psf_template: float,
            scaling_factor: float = 1,
            checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        """
        Constructor of the class. The Contrast class can be used with and
        without a checkpoint_dir. If a checkpoint_dir is given intermediate
        results (such as residuals with fake planets) can be restored.

        Args:
            science_sequence: A 3d numpy array of the observation
                sequence without any fake planets.
            psf_template: A 2d numpy array with the psf-template
                (usually the unsaturated star).
            psf_fwhm_radius: The FWHM (radius) of the PSF. It is needed to
                sample independent noise values i.e. it determines the
                spacing between the noise observations which are extracted
                from the residuals.
            parang_rad: A 1d numpy array containing the parallactic angles
                in radians.
            dit_science: Integration time of the science frames.
            dit_psf_template: Integration time of the psf_template.
            scaling_factor: A scaling factor to account for e.g. ND filters.
            checkpoint_dir: A directory in which intermediate results are
                stored. Within the checkpoint_dir three subdirectories are
                created:

                    1. configs_cgrid: This folder will contain several
                    .json config files which define where to insert fake
                    planets. The config files are created within
                    :meth:`~design_fake_planet_experiments`.

                    2. residuals: This folder contains the results of the
                    post-processing algorithm used. E.g. PCA residuals.
                    The results are calculated in
                    :meth:`~run_fake_planet_experiments`.

                    3. scratch: A scratch folder which can be used by the
                    post-processing algorithm to store files.
        """

        self.science_sequence = science_sequence
        self.psf_template = psf_template
        self.parang_rad = parang_rad
        self.dit_science = dit_science
        self.dit_psf_template = dit_psf_template
        self.scaling_factor = scaling_factor

        # create structure for the checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        sub_folders = create_checkpoint_folders(self.checkpoint_dir)
        self.config_dir, self.residual_dir, self.scratch_dir = sub_folders

        self.psf_fwhm_radius = psf_fwhm_radius

        # Members which are created later
        self.experimental_setups = None
        self.results_dict = None
        self.stellar_flux = None
        self.contrast_results = None

    @classmethod
    def create_from_checkpoint_dir(
            cls,
            psf_template: np.ndarray,
            psf_fwhm_radius: float,
            dit_science: float,
            dit_psf_template: float,
            checkpoint_dir: Union[str, Path],
            scaling_factor: float = 1) -> Contrast:
        """
        A factory method which can be used to restore a Contrast instance from
        a checkpoint_dir. This function can be used to load results calculated
        earlier with :meth:`~run_fake_planet_experiments`. It is further useful
        to load results of more complex post-processing methods computed on
        a cluster.

        Args:
            psf_template: A 2d numpy array with the psf-template
                (usually the unsaturated star).
            psf_fwhm_radius: The FWHM (radius) of the PSF. It is needed to
                sample independent noise values i.e. it determines the
                spacing between the noise observations which are extracted
                from the residuals.
            dit_science: Integration time of the science frames.
            dit_psf_template: Integration time of the psf_template.
            checkpoint_dir: The directory in which intermediate results are
                stored. It has to contain three subdirectories:

                    1. configs_cgrid: This folder contains several .json config
                    files which define where fake planets have been inserted.

                    2. residuals: This folder contains the results of the
                    post-processing algorithm used. E.g. PCA residuals.

                    3. scratch: A scratch folder which can be used by the
                    post-processing algorithm to store files.

            scaling_factor: A scaling factor to account for e.g. ND filters.

        Returns:
            An instance of Contrast ready to run
            :meth:`~prepare_contrast_results`
            :meth:`~compute_analytic_contrast_curves` and
            :meth:`~compute_contrast_grids`.
        """

        # 1.) Create an instance of Contrast
        contrast_instance = cls(
            science_sequence=np.empty(0),
            psf_template=psf_template,
            psf_fwhm_radius=psf_fwhm_radius,
            parang_rad=np.empty(0),
            dit_science=dit_science,
            dit_psf_template=dit_psf_template,
            scaling_factor=scaling_factor,
            checkpoint_dir=checkpoint_dir)

        # Note: we don't have to restore the experimental_setups as they are not
        # used by compute_analytic_contrast_curves or compute_contrast_grids

        # 2.) restore the results_dict from the checkpoint_dir
        # we have to this for each method directory we find
        methods = []
        for tmp_folder in contrast_instance.residual_dir.iterdir():
            if not tmp_folder.is_dir():
                continue

            methods.append(tmp_folder)

        # create an empty dict for the result we read
        contrast_instance.results_dict = dict()

        for tmp_method_dir in sorted(methods):
            # search for all files for the given method
            result_files = search_for_config_and_residual_files(
                config_dir=contrast_instance.config_dir,
                method_dir=tmp_method_dir)

            # restore the results
            tmp_results = read_fake_planet_results(result_files)
            contrast_instance.results_dict[tmp_method_dir.name] = tmp_results

        return contrast_instance

    def design_fake_planet_experiments(
            self,
            flux_ratios: np.ndarray,
            num_planets: int = 6,
            separations: Optional[np.ndarray] = None,
            overwrite: bool = False
    ) -> None:
        """
        Calculates the positions at which fake planets are inserted. For each
        fake planet experiment one .json config file is created (in case a
        checkpoint_dir is available). This function is the first step to
        calculate a
        `contrast curve <../02_user_documentation/01_contrast_curves.ipynb#Step-1:-Design-fake-planet-experiments>`_
        or
        `contrast grid <../02_user_documentation/02_contrast_grid.ipynb>`_.

        Args:
            flux_ratios: A list / single value of planet-to-star
                flux_ratios used for the fake planets to be injected.
                If you want to calculate a simple contrast curve this values
                should be below the expected detection limit.
                For the computation of a contrast grid several flux_ratios are
                needed.
            num_planets: The number of planets to be inserted. Has to be
                between 1 (minimum) and 6 (maximum). More planets result in more
                accurate results but also longer computation time.
            separations: Separations at which fake planets are inserted [pixel].
                By default, (If set to None) separations are selected in steps
                of 1 FWHM form the central star to the edge of the image.
            overwrite: Check if config files exist already within the
                checkpoint_dir. Overwrite allows to overwrite already existing
                files. The default behaviour will raise an error.

        """

        # 1. Calculate test positions for the fake planets
        # Take the first image of the science_sequence as a test_image
        test_image = self.science_sequence[0]

        planet_positions = calculate_fake_planet_positions(
            test_img=test_image,
            psf_fwhm_radius=self.psf_fwhm_radius,
            num_planets=num_planets,
            separations=separations)

        # 2. generate all experiments
        if isinstance(flux_ratios, float):
            flux_ratios = [flux_ratios, ]

        self.experimental_setups = generate_fake_planet_experiments(
            flux_ratios=flux_ratios,
            planet_positions=planet_positions)

        # 3. save the config files if requested
        if self.config_dir is not None:
            save_experiment_configs(
                experimental_setups=self.experimental_setups,
                experiment_config_dir=self.config_dir,
                overwrite=overwrite)

    def _check_residuals_exist_and_restore(
            self,
            algorithm_function: DataReductionInterface,
            exp_id: str
    ) -> Union[bool, Dict[str, np.ndarray]]:
        """
        Checks the residual directory for already existing residuals. If all
        residuals needed exist already they are restored and returned.

        Args:
            algorithm_function: The post-processing method used to calculate
                the residuals (e.g. PCA).
            exp_id: Experiment ID of the config used to add the fake
                planet.

        Returns:
            False if residuals to not exist. Returns a dict with the restored
            residuals if all residuals are found.

        """

        method_keys = algorithm_function.get_method_keys()
        result_dict = dict()

        for tmp_method_key in method_keys:
            result_dict[tmp_method_key] = dict()
            tmp_sub_dir = self.residual_dir / tmp_method_key

            # if the subdir does not exist at all no residual exist either
            if not tmp_sub_dir.is_dir():
                return False

            # check if the residual with the given tmp_method_key exists
            exp_name = "residual_ID_" + exp_id + ".fits"
            tmp_file = tmp_sub_dir / exp_name

            # if it does not exist we can return false
            if not tmp_file.is_file():
                return False

            # restore the residual
            result_dict[tmp_method_key] = open_fits(tmp_file)

        # All residuals exist
        return result_dict

    def _run_fake_planet_experiment(
            self,
            algorithm_function: DataReductionInterface,
            exp_id: str
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """
        Runs a single fake planet experiment by adding a fake planet to the
        planet free sequence, running the post-processing algorithm and save
        the results as .fits (if a checkpoint_dir is available).
        _run_fake_planet_experiment is used for the multiprocessing
        in :meth:`~run_fake_planet_experiments`.

        Args:
            algorithm_function: The post-processing method used to calculate
                the residuals (e.g. PCA). See `wrappers <wrappers.html>`_
                for an examples.
            exp_id:  Experiment ID of the config used to add the fake
                planet.

        Returns: The exp_id and finished residual as a tuple.

        """

        experimental_setup = self.experimental_setups[exp_id]

        # 1.) Check if the expected residuals already exist
        if self.residual_dir is not None:
            restored_residuals = self._check_residuals_exist_and_restore(
                algorithm_function,
                exp_id)

            # if yes use the restored_residuals
            if restored_residuals:
                return exp_id, restored_residuals

        # if not run the fake planet experiment

        # 2.) create the fake planet stack
        stack_with_fake_planet = add_fake_planets(
            input_stack=self.science_sequence,
            psf_template=self.psf_template,
            parang=self.parang_rad,
            dit_science=self.dit_science,
            dit_psf_template=self.dit_psf_template,
            experiment_config=experimental_setup,
            scaling_factor=self.scaling_factor)

        # 3.) Compute the residuals
        residuals = algorithm_function(
            stack_with_fake_planet,
            self.parang_rad,
            self.psf_template,
            exp_id)

        # 4.) Save the result if needed
        if self.residual_dir is None:
            return exp_id, residuals

        # for each method and residual pair
        for tmp_method_key, tmp_residual in residuals.items():

            # Create a subdirectory for the method if it does not exist
            tmp_sub_dir = self.residual_dir / tmp_method_key
            if not tmp_sub_dir.is_dir():
                tmp_sub_dir.mkdir()

            # Save the residual as a .fits file
            exp_name = "residual_ID_" + exp_id + ".fits"
            tmp_file = tmp_sub_dir / exp_name

            if not tmp_file.is_file():
                save_as_fits(tmp_residual, tmp_file)

        return exp_id, residuals

    def run_fake_planet_experiments(
            self,
            algorithm_function: DataReductionInterface,
            num_parallel: int
    ) -> None:
        """
        Runs the fake planet experiments with the  post-processing algorithm
        given as algorithm_function. Requires that fake planet experiments have
        been defined with :meth:`~design_fake_planet_experiments` before.
        For each fake planet experiment the following steps are executed:

            1. Insert a fake planet in the observation sequence.

            2. Run algorithm_function to get a residual.

            3. Save the residual in case a checkpoint_dir is available.

        All fake planet experiments are executed with multiprocessing.
        **This function is the second step to calculate a contrast curve
        or contrast grid.**

        Args:
            algorithm_function: The post-processing method used to calculate
                the residuals (e.g. PCA). See `wrappers <wrappers.html>`_
                for examples.
            num_parallel: The number of parallel fake planet experiments.

        """

        # 1. Run the data reduction in parallel
        # The _run_fake_planet_experiment checks if residuals already exist
        # and only computes the missing ones
        print("Running fake planet experiments...", end="")
        results = Parallel(n_jobs=num_parallel)(
            delayed(self._run_fake_planet_experiment)(
                algorithm_function,
                i) for i in tqdm(self.experimental_setups))
        tmp_results_dict = dict(results)
        print("[DONE]")

        # 2. Prepare the results for the ContrastResult
        # Invert the dict structure. We want the method names as keys.
        # For each method we want a list containing tuples with the experiment
        # setup and residual
        self.results_dict = dict()

        for fake_planet_id, value in tmp_results_dict.items():
            for method_key, tmp_residual in value.items():

                if method_key not in self.results_dict:
                    self.results_dict[method_key] = list()

                tmp_config = self.experimental_setups[fake_planet_id]
                self.results_dict[method_key].append(
                    (tmp_config, tmp_residual))

    def prepare_contrast_results(
            self,
            photometry_mode_planet: AperturePhotometryMode,
            photometry_mode_noise: AperturePhotometryMode
    ) -> None:
        """
        After running :meth:`~run_fake_planet_experiments` this function is used
        to compute the stellar flux and prepare instances of
        :meth:`~applefy.detections.evaluation.ContrastResult`.
        **This function is the third step to calculate a contrast curve
        or contrast grid.**

        Args:
            photometry_mode_planet: An instance of AperturePhotometryMode which
                defines how the flux is measured at the planet position.
            photometry_mode_noise: An instance of AperturePhotometryMode which
                defines how the noise photometry is measured.
        """

        # 0.) Check if run_fake_planet_experiments was executed before
        if self.results_dict is None:
            raise RuntimeError(
                "prepare_contrast_results requires that "
                "run_fake_planet_experiments was executed before.")

        # 1.) Estimate the stellar flux
        self.stellar_flux = estimate_stellar_flux(
            psf_template=self.psf_template,
            dit_science=self.dit_science,
            dit_psf_template=self.dit_psf_template,
            photometry_mode=photometry_mode_planet,
            scaling_factor=self.scaling_factor)

        # 2.) For each method setup create one ContrastResult
        self.contrast_results = dict()
        for tmp_method_name, method_results in self.results_dict.items():
            tmp_contrast_result = ContrastResult(
                model_results=method_results,
                stellar_flux=self.stellar_flux,
                photometry_mode_planet=photometry_mode_planet,
                photometry_mode_noise=photometry_mode_noise,
                psf_fwhm_radius=self.psf_fwhm_radius)

            self.contrast_results[tmp_method_name] = tmp_contrast_result

    def _get_result_table_index(
            self,
            pixel_scale: Optional[float] = None
    ) -> pd.MultiIndex:
        """
        A simple helper function to contrast the pandas MultiIndex needed in
        :meth:`~compute_analytic_contrast_curves` and
        :meth:`~compute_contrast_grids`.

        Args:
            pixel_scale: The pixel scale in arcsec.

        Returns: The pandas MultiIndex with separations in FWHM as well as
            arcsec.
        """

        example_contrast_result = next(iter(self.contrast_results.values()))
        separations_fwhm = \
            [i / (self.psf_fwhm_radius * 2)
             for i in example_contrast_result.idx_table.index]

        # create the index in FWHM and arcsec
        if pixel_scale is None:
            separation_index = pd.Index(
                separations_fwhm,
                name=r"separation [FWHM]")

        else:
            separations_arcsec = \
                [i * pixel_scale
                 for i in example_contrast_result.idx_table.index]

            separation_index = pd.MultiIndex.from_tuples(
                list(zip(separations_fwhm, separations_arcsec)),
                names=[r"separation [$FWHM$]",
                       "separation [arcsec]"])

        return separation_index

    def compute_analytic_contrast_curves(
            self,
            statistical_test: TestInterface,
            confidence_level_fpf: float,
            num_rot_iter: int = 20,
            pixel_scale: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes
        `analytic contrast curves <../02_user_documentation/01_contrast_curves.ipynb>`_
        given a confidence level and a
        statistical test. Analytic contrast curves are only
        applicable if used with linear post-processing techniques such as PCA.
        They can further lead to inaccurate results close to the star. For more
        advanced post-processing techniques use a contrast grid instead.

        Requires a previous execution of :meth:`~prepare_contrast_results`.
        If the post-processing method used in
        :meth:`~run_fake_planet_experiments` returns multiple residuals
        (e.g. for different numbers of PCA components) the output contains
        multiple contrast curves summarized in one pandas table.

        Args:
            statistical_test: The test used to constrain the planet flux
                needed to be counted as a detection.
                For the classical TTest (Gaussian noise) use an instance of
                :meth:`~applefy.statistics.parametric.TTest`. For Laplacian
                noise use
                :meth:`~applefy.statistics.bootstrapping.LaplaceBootstrapTest`.
            confidence_level_fpf: The confidence level associated with the
                contrast curve as false-positive fraction (FPF).
            num_rot_iter: Number of tests performed with different positions of
                the noise values. See
                `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
                for more information.
            pixel_scale: The pixel scale in arcsec. If given the result table
                will have a multi index.

        Returns:
            1. A pandas DataFrame with the median contrast curves over all
            num_rot_iter.

            2. A pandas DataFrame with the MAD error of the contrast curves over
            all  num_rot_iter.
        """

        # 0.) Check if prepare_contrast_results was executed before
        if self.contrast_results is None:
            raise RuntimeError(
                "compute_contrast_grids requires that "
                "prepare_contrast_results was executed before.")

        # 1.) Compute contrast curves for all method configurations
        contrast_curves = dict()
        contrast_curves_mad = dict()

        for key, tmp_result in self.contrast_results.items():
            print("Computing contrast curve for " + str(key))
            tmp_contrast_curve, tmp_contrast_curve_error = \
                tmp_result.compute_analytic_contrast_curve(
                    statistical_test=statistical_test,
                    num_rot_iter=num_rot_iter,
                    confidence_level_fpf=confidence_level_fpf)

            contrast_curves[key] = tmp_contrast_curve["contrast"].values
            contrast_curves_mad[key] = \
                tmp_contrast_curve_error["MAD of contrast"].values

        # 2.) Merge the results into two pandas table
        separation_index = self._get_result_table_index(pixel_scale)

        # create the final tables
        pd_contrast_curves = pd.DataFrame(
            contrast_curves, index=separation_index).replace(
            [np.inf, -np.inf], np.inf)

        pd_contrast_curves_mad = pd.DataFrame(
            contrast_curves_mad, index=separation_index).replace(
            [np.inf, -np.inf], np.inf)

        return pd_contrast_curves, pd_contrast_curves_mad

    def compute_contrast_grids(
            self,
            statistical_test: TestInterface,
            confidence_level_fpf: float,
            num_cores: int = 1,
            safety_margin: float = 1.0,
            num_rot_iter: int = 20,
            pixel_scale: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculates the
        `contrast grids <../02_user_documentation/02_contrast_grid.ipynb>`_.
        A contrast grid shows the detection uncertainty as a function of
        separation from the star and fake planet flux_ratio. It evaluates the
        fake planet residuals directly.

        Requires a previous execution of :meth:`~prepare_contrast_results`.
        If the post-processing method used in
        :meth:`~run_fake_planet_experiments` returns multiple residuals
        (e.g. for different numbers of PCA components) the output contains
        multiple contrast grids. The results are computed using multiprocessing.

        Args:
            statistical_test: The test used to constrain the planet flux
                needed in order to be counted as a detection.
                For the classical TTest (Gaussian noise) use an instance of
                :meth:`~applefy.statistics.parametric.TTest`. For Laplacian
                noise use
                :meth:`~applefy.statistics.bootstrapping.LaplaceBootstrapTest`.
            num_cores: Number of parallel jobs used during multiprocessing.
            confidence_level_fpf: The confidence level associated with the
                contrast curve as false-positive fraction (FPF).
                See return values.
            safety_margin: Area around the planet [pixel] which is excluded from
                the noise. This can be useful in case the planet has negative
                wings.
            num_rot_iter: Number of tests performed with different positions of
                the noise values. See
                `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
                for more information.
            pixel_scale: The pixel scale in arcsec. If given the result table
                will have a multi index.

        Returns:
            1. A pandas DataFrame with the contrast curves obtained by
            thresholding the contrast grids. We report the median
            p-values over all num_rot_iter experiments performed.

            2. A dict with one contrast grid for each output of the
            post-processing routine. We report the median
            p-value over all num_rot_iter experiments performed.
        """

        # 0.) Check if prepare_contrast_results was executed before
        if self.contrast_results is None:
            raise RuntimeError(
                "compute_contrast_grids requires that "
                "prepare_contrast_results was executed before.")

        # 1.) Compute contrast grids for all method configurations
        contrast_curves = dict()
        contrast_grids = dict()

        for key, tmp_result in self.contrast_results.items():
            print("Computing contrast grid for " + str(key))

            tmp_contrast_grid, tmp_contrast_grid_curve = \
                tmp_result.compute_contrast_grid(
                    statistical_test=statistical_test,
                    num_cores=num_cores,
                    num_rot_iter=num_rot_iter,
                    safety_margin=safety_margin,
                    confidence_level_fpf=confidence_level_fpf)

            # Convert the separation index to FWHM
            tmp_contrast_grid.columns = self._get_result_table_index()
            contrast_curves[key] = tmp_contrast_grid_curve["contrast"].values
            contrast_grids[key] = tmp_contrast_grid

        # 2.) merge the results of the contrast_curves into one nice table
        separation_index = self._get_result_table_index(pixel_scale)

        # create the final tables
        pd_contrast_curves = pd.DataFrame(
            contrast_curves, index=separation_index).replace(
            [np.inf, -np.inf], np.inf)

        return pd_contrast_curves, contrast_grids


class ContrastResult:
    """
    Class for the evaluation and organization of residuals from one
    method (e.g. pca with 10 components). Supports both contrast curves and
    contrast grids. Usually ContrastResult is used within an instance of
    :meth:`~Contrast`. But, it can also be used individually.
    """

    def __init__(
            self,
            model_results: List[Tuple[Dict[str, Any], np.ndarray]],
            stellar_flux: float,
            photometry_mode_planet: AperturePhotometryMode,
            photometry_mode_noise: AperturePhotometryMode,
            psf_fwhm_radius: float):
        """
        Constructor of the class. This function will sort all residuals.

        Args:
            model_results: List which contains tuples of fake planet config
                files (as dict) and residuals as created by
                :meth:`~applefy.utils.file_handling.read_fake_planet_results`.
            stellar_flux: The stellar flux measured with
                :meth:`~applefy.utils.photometry.estimate_stellar_flux`.
                The mode used to get the stellar flux has to be the same as used
                in planet_photometry_mode.
            photometry_mode_planet: An instance of AperturePhotometryMode which
                defines how the flux is measured at the planet position.
            photometry_mode_noise: An instance of AperturePhotometryMode which
                defines how the noise photometry is measured.
            psf_fwhm_radius: The FWHM (radius) of the PSF. It is needed to
                sample independent noise values i.e. it determines the
                spacing between the noise observations which are extracted
                from the residuals.
        """

        # Init additional members needed for flux based estimations
        self.stellar_flux = stellar_flux

        # Check if photometry_modes are compatible
        if not photometry_mode_planet.check_compatible(photometry_mode_noise):
            raise ValueError("Photometry modes " +
                             photometry_mode_planet.flux_mode + " and " +
                             photometry_mode_noise.flux_mode + " are not"
                             " compatible.")

        # Save the inputs
        self.psf_fwhm_radius = psf_fwhm_radius
        self.planet_mode = photometry_mode_planet
        self.noise_mode = photometry_mode_noise

        # Read in the results
        read_in = sort_fake_planet_results(model_results)
        self.fp_residual, self.planet_dict, self.idx_table = read_in

        # In case throughput values are computed later we initialize the member
        # variables here
        self.throughput_dict = None
        self.median_throughput_table = None

    def compute_throughput(self) -> pd.DataFrame:
        """
        Computes a throughput table based on all residuals. Uses the function
        :meth:`~applefy.utils.throughput.compute_throughput_table`.

        Returns:
            Pandas DataFrame which contains the median throughput
            as a function of separation and fake planet flux_ratio.
        """

        if self.median_throughput_table is not None:
            return self.median_throughput_table

        self.throughput_dict, self.median_throughput_table = \
            compute_throughput_table(self.planet_dict,
                                     self.fp_residual,
                                     self.idx_table,
                                     self.stellar_flux,
                                     photometry_mode_planet=self.planet_mode)

        return self.median_throughput_table

    @property
    def residuals(self) -> np.ndarray:
        """
        Combines all residuals from all experiments.

        Returns:
            All residuals from all experiments as one array with dimensions

                (separation, fake planet flux ratio, planet index, x, y)
        """

        return merge_fake_planet_residuals(self.planet_dict,
                                           self.idx_table)

    def compute_analytic_contrast_curve(
            self,
            statistical_test: TestInterface,
            confidence_level_fpf: Union[float, List[float]],
            num_rot_iter: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes an
        `analytic contrast curve <../02_user_documentation/01_contrast_curves.ipynb>`_
        given a confidence level and a statistical test.
        Analytic contrast curves are only applicable if used with linear
        post-processing techniques such as PCA.
        They can further lead to inaccurate results close to the star. For more
        advanced post-processing techniques use a contrast grid instead.

        Args:

            statistical_test: The test used to constrain the planet flux
                needed to be counted as a detection.
                For the classical TTest (Gaussian noise) use an instance of
                :meth:`~applefy.statistics.parametric.TTest`. For Laplacian
                noise use
                :meth:`~applefy.statistics.bootstrapping.LaplaceBootstrapTest`.
            confidence_level_fpf: The confidence level associated with the
                contrast curve as false-positive fraction (FPF). Can also be
                a list of fpf values (with the length of separations evaluated
                in the fake planet experiments).
            num_rot_iter: Number of tests performed with different positions of
                the noise values. See
                `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
                for more information.

        Returns:
            1. A pandas DataFrame with the median contrast curve over all
            num_rot_iter.

            2. A pandas DataFrame with the MAD error of the contrast curve over
            all  num_rot_iter.
        """

        if self.median_throughput_table is None:
            self.compute_throughput()

        median_contrast_curve, contrast_error, _ = compute_contrast_curve(
            # use the last row of the throughput table as the throughput
            throughput_list=self.median_throughput_table.T.iloc[-1],
            stellar_flux=self.stellar_flux,
            fp_residual=self.fp_residual,
            confidence_level_fpf=confidence_level_fpf,
            statistical_test=statistical_test,
            psf_fwhm_radius=self.psf_fwhm_radius,
            photometry_mode_noise=self.noise_mode,
            num_rot_iter=num_rot_iter)

        # wrap contrast curves into pandas arrays
        median_contrast_curve = pd.DataFrame(
            median_contrast_curve,
            index=self.idx_table.index,
            columns=["contrast", ])

        contrast_error = pd.DataFrame(
            contrast_error,
            index=self.idx_table.index,
            columns=["MAD of contrast", ])

        return median_contrast_curve, contrast_error

    def compute_contrast_grid(
            self,
            statistical_test: TestInterface,
            num_cores: int = 1,
            num_rot_iter: int = 20,
            safety_margin: float = 1.0,
            confidence_level_fpf: Optional[float] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Calculates the
        `contrast grid  <../02_user_documentation/02_contrast_grid.ipynb>`_ .
        The contrast grid shows the detection uncertainty as a function of
        separation from the star and fake planet flux_ratio. It evaluates the
        fake planet residuals directly.
        Compared to the function :meth:`~compute_analytic_contrast_curve` this
        function is applicable not only to residuals of linear PSF-subtraction
        methods like PCA, but in general.
        This allows to compare results of different methods.
        The results are computed using multiprocessing.

        Args:
            statistical_test: The test used to constrain the planet flux
                needed in order to be counted as a detection.
                For the classical TTest (Gaussian noise) use an instance of
                :meth:`~applefy.statistics.parametric.TTest`. For Laplacian
                noise use
                :meth:`~applefy.statistics.bootstrapping.LaplaceBootstrapTest`.
            num_cores: Number of parallel jobs used during multiprocessing.
            num_rot_iter: Number of tests performed with different positions of
                the noise values. See
                `Figure 02 <../04_apples_with_apples/paper_experiments/02_Rotation.ipynb>`_
                for more information.
            safety_margin: Area around the planet [pixel] which is excluded from
                the noise. This can be useful in case the planet has negative
                wings.
            confidence_level_fpf: If set to a float value the output contrast
                grid will be interpolated in order to obtain a contrast curve.
                The value is the confidence level associated with the
                contrast curve as false-positive fraction (FPF).
                If None only the contrast grid is returned.

        Returns:
            1. A pandas DataFrame with the contrast grid. We report the median
            p-value over all num_rot_iter experiments performed.

            2. A pandas DataFrame with the contrast curve obtained by
            thresholding the contrast grid. We report the median
            p-values over all num_rot_iter experiments performed. Only returned
            if a confidence_level_fpf is given.

        """

        contrast_grid = compute_contrast_grid(
            planet_dict=self.planet_dict,
            idx_table=self.idx_table,
            statistical_test=statistical_test,
            psf_fwhm_radius=self.psf_fwhm_radius,
            photometry_mode_planet=self.planet_mode,
            photometry_mode_noise=self.noise_mode,
            num_cores=num_cores,
            num_rot_iter=num_rot_iter,
            safety_margin=safety_margin)

        if isinstance(confidence_level_fpf, (float, np.floating)):
            # compute the contrast curve
            contrast_curve = compute_contrast_from_grid(
                contrast_grid,
                confidence_level_fpf)

            return contrast_grid, contrast_curve

        return contrast_grid


class DataReductionInterface(ABC):
    """
    Applefy allows to compute contrast curves with different post-processing
    algorithms. However, it does not come with any implementation of these
    techniques. Instead, we use existing implementations in packages like
    `PynPoint <https://pynpoint.readthedocs.io/en/latest/>`_ or
    `VIP <https://vip.readthedocs.io/en/latest/>`_.
    The DataReductionInterface is an interface which guarantees that these
    external implementations can be used within applefy.
    See `wrappers <wrappers.html>`_ for examples.
    """

    @abstractmethod
    def get_method_keys(self) -> List[str]:
        """
        The get_method_keys should return the name (or names) of the method
        implemented. The name(s) should match the keys of the result dict
        created by __call__.

        Returns:
            A list containing the name(s) of the method. If __call__ computes
            only a single residual the return value should be a list containing
            a single string. If __call__ has multiple outputs (e.g. PCA
            residuals with different number of components) the list should
            contain one name for each entry in the result dict of __call__.
        """

    @abstractmethod
    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:
        """
        In order to make external post-processing methods compatible with
        applefy they have to implement the __call__ function. __call__ should
        run the post-processing algorithm and return its result(s) /
        residual(s). It is possible to return multiple result (e.g. for
        different parameters of the post-processing algorithm) as a dict.

        Args:
             stack_with_fake_planet: A 3d numpy array of the observation
                sequence. Fake plants are inserted by applefy in advance.
             parang_rad: A 1d numpy array containing the parallactic angles
                in radians.
             psf_template: A 2d numpy array with the psf-template
                (usually the unsaturated star).
             exp_id: Experiment ID of the config used to add the fake
                planet. It is a unique string and can be used to store
                intermediate results. See :meth:`~applefy.detections.\
preparation.generate_fake_planet_experiments` for more information about the
                config files.

        Returns:
            A dictionary which contains the results / residuals of the
            post-processing algorithm. Each entry in the dict should give one
            result, while the key specifies the name of the method used. If
            e.g. PCA with different number of components is used the dict
            contains one residual for each number of components. The keys of
            the dict have to match those specified in :meth:`get_method_keys`.

        """
