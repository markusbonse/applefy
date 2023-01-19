from pathlib import Path
from joblib import Parallel, delayed
from abc import ABC, abstractmethod

from applefy.utils.data_handling import save_as_fits, open_fits
from applefy.detections.preparation import calculate_planet_positions, \
    generate_fake_planet_experiments, save_experiment_configs
from applefy.detections.execution import add_fake_planets
from applefy.detections.evaluation import estimate_stellar_flux, ContrastResult


class DataReductionInterface(ABC):

    @abstractmethod
    def get_method_keys(self):
        pass

    @abstractmethod
    def __call__(
            self,
            stack_with_fake_planet,
            parang_rad,
            psf_template,
            exp_id):
        pass


class Contrast:

    def __init__(
            self,
            science_sequence,
            psf_template,
            psf_fwhm_radius,
            parang,
            dit_science,
            dit_psf_template,
            scaling_factor=1,
            checkpoint_dir=None):

        self.science_sequence = science_sequence
        self.psf_template = psf_template
        self.parang = parang
        self.dit_science = dit_science
        self.dit_psf_template = dit_psf_template
        self.scaling_factor = scaling_factor

        # create structure for the checkpoints
        self.checkpoint_dir = checkpoint_dir

        sub_folders = self._create_checkpoint_folders()
        self.config_dir, self.residual_dir, self.scratch_dir = sub_folders

        # TODO add auto mode
        self.psf_fwhm_radius = psf_fwhm_radius

        # Members which are created later
        self.experimental_setups = None
        self.results_dict = None
        self.stellar_flux = None
        self.contrast_results = None

    @classmethod
    def create_from_checkpoint_dir(
            cls,
            checkpoint_dir):
        pass

    def _create_checkpoint_folders(self):
        self.config_dir = None
        self.residuals_dir = None

        # if no experiment_root_dir is given we don't save results
        if self.checkpoint_dir is None:
            return

        # use pathlib for easy path handling
        self.checkpoint_dir = Path(self.checkpoint_dir)

        # check if the experiment_root_dir exists
        if not self.checkpoint_dir.is_dir():
            raise IOError("The directory " + str(self.checkpoint_dir) +
                          " does not exist. Please create it.")

        # create sub-folders if they do not exist
        config_dir = self.checkpoint_dir / "configs_cgrid"
        residual_dir = self.checkpoint_dir / "residuals"
        scratch_dir = self.checkpoint_dir / "scratch"

        config_dir.mkdir(parents=False, exist_ok=True)
        residual_dir.mkdir(parents=False, exist_ok=True)
        scratch_dir.mkdir(parents=False, exist_ok=True)

        return config_dir, residual_dir, scratch_dir

    def design_fake_planet_experiments(
            self,
            flux_ratios,
            num_planets=6,
            separations=None,
            overwrite=False):

        # 1. Calculate test positions for the fake planets
        # Take the first image of the science_sequence as a test_image
        test_image = self.science_sequence[0]

        planet_positions = calculate_planet_positions(
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
            algorithm_function,
            fake_planet_id):

        method_keys = algorithm_function.get_method_keys()
        result_dict = dict()

        for tmp_method_key in method_keys:
            result_dict[tmp_method_key] = dict()
            tmp_sub_dir = self.residual_dir / tmp_method_key

            # if the subdir does not exist at all no residual exist either
            if not tmp_sub_dir.is_dir():
                return False

            # check if the residual with the given tmp_method_key exists
            exp_name = "residual_ID_" + fake_planet_id + ".fits"
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
            exp_id):

        experimental_setup = self.experimental_setups[exp_id]

        # 1.) Check if the expected residuals already exist
        if self.residual_dir is not None:
            restored_residuals = self._check_residuals_exist_and_restore(
                algorithm_function,
                exp_id)

            # if yes use the restored_residuals
            if restored_residuals:
                print("Found all residuals for experiment ID: " + exp_id)
                return exp_id, restored_residuals

        # if not run the fake planet experiment

        # 2.) create the fake planet stack
        stack_with_fake_planet = add_fake_planets(
            input_stack=self.science_sequence,
            psf_template=self.psf_template,
            parang=self.parang,
            dit_science=self.dit_science,
            dit_psf_template=self.dit_psf_template,
            experiment_config=experimental_setup,
            scaling_factor=self.scaling_factor)

        # 3.) Compute the residuals
        residuals = algorithm_function(
            stack_with_fake_planet,
            self.parang,
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
            algorithm_function,
            num_parallel):

        # 1. Run the data reduction in parallel
        # The _run_fake_planet_experiment checks if residuals already exist
        # and only computes the missing ones
        results = Parallel(n_jobs=num_parallel)(
            delayed(self._run_fake_planet_experiment)(
                algorithm_function,
                i) for i in self.experimental_setups.keys())
        tmp_results_dict = dict(results)

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
            photometry_mode_planet,
            photometry_mode_noise,
            scaling_factor=1.):

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
            scaling_factor=scaling_factor)

        # 2.) For each method setup create one ContrastResult
        self.contrast_results = dict()
        for tmp_method_name, method_results in self.results_dict.items():
            tmp_contrast_result = ContrastResult(
                model_results_in=method_results,
                stellar_flux=self.stellar_flux,
                planet_photometry_mode=photometry_mode_planet,
                noise_photometry_mode=photometry_mode_noise,
                psf_fwhm_radius=self.psf_fwhm_radius)

            self.contrast_results[tmp_method_name] = tmp_contrast_result
