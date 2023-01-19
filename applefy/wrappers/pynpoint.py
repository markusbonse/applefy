import shutil
import h5py
import numpy as np
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint import Pypeline, PcaPsfSubtractionModule

from applefy.detections.contrast import DataReductionInterface


class SimplePCAPynPoint(DataReductionInterface):

    def __init__(self, num_pca):
        self.num_pca = num_pca

    def get_method_keys(self):
        return ["PCA (" + str(self.num_pca).zfill(3) + " components)", ]

    def __call__(
            self,
            stack_with_fake_planet,
            parang_rad,
            psf_template,
            exp_id):

        _, residual_stack_fake = pca_psf_subtraction(
            images=stack_with_fake_planet,
            angles=-np.rad2deg(parang_rad),
            pca_number=self.num_pca)

        # Average along temporal axis
        residual_image = np.mean(residual_stack_fake, axis=0)

        result_dict = dict()
        result_dict["PCA (" + str(self.num_pca).zfill(3) + " components)"] = \
            residual_image

        return result_dict


class MultiComponentPCAPynPoint(DataReductionInterface):

    def __init__(
            self,
            num_pcas,
            scratch_dir,
            num_cpus_pynpoint):

        self.num_pcas = num_pcas
        self.scratch_dir = scratch_dir
        self.num_cpus_pynpoint = num_cpus_pynpoint

    def get_method_keys(self):
        keys = ["PCA (" + str(num_pcas).zfill(3) + " components)"
                for num_pcas in self.num_pcas]

        return keys

    def __call__(
            self,
            stack_with_fake_planet,
            parang_rad,
            psf_template,
            exp_id):

        pynpoint_dir = "tmp_pynpoint_" + exp_id
        pynpoint_dir = self.scratch_dir / pynpoint_dir

        if not pynpoint_dir.is_dir():
            pynpoint_dir.mkdir()

        out_file = h5py.File(
            pynpoint_dir / "PynPoint_database.hdf5",
            mode='w')

        out_file.create_dataset("data_with_planet", data=stack_with_fake_planet)
        out_file.create_dataset("header_data_with_planet/PARANG",
                                data=np.rad2deg(parang_rad))
        out_file.close()

        # 6.) Create PynPoint Pipeline and run PCA
        pipeline = Pypeline(working_place_in=str(pynpoint_dir),
                            input_place_in=str(pynpoint_dir),
                            output_place_in=str(pynpoint_dir))

        pipeline.set_attribute("config", "CPU",
                               attr_value=self.num_cpus_pynpoint)

        pca_subtraction = PcaPsfSubtractionModule(
            name_in="pca_subtraction",
            images_in_tag="data_with_planet",
            reference_in_tag="data_with_planet",
            res_mean_tag="residuals_out_mean",
            res_median_tag=None,
            res_weighted_tag=None,
            pca_numbers=self.num_pcas,
            processing_type="ADI")

        pipeline.add_module(pca_subtraction)
        pipeline.run_module("pca_subtraction")

        # 7.) Get the data from the Pynpoint database
        result_dict = dict()

        residuals = pipeline.get_data("residuals_out_mean")
        for idx, tmp_algo_name in enumerate(self.get_method_keys()):
            result_dict[tmp_algo_name] = residuals[idx]

        # Delete the temporary database
        shutil.rmtree(pynpoint_dir)

        return result_dict
