"""
The following wrapper classes
allow to use `PynPoint <https://pynpoint.readthedocs.io/en/latest/>`__ with
applefy. PynPoint is not on the requirement list of
applefy. It has to be installed separately.
"""

import sys, os
import shutil
import warnings
from typing import List, Dict
from pathlib import Path

import h5py
import numpy as np
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint import Pypeline, PcaPsfSubtractionModule

from applefy.detections.contrast import DataReductionInterface


class SimplePCAPynPoint(DataReductionInterface):
    """
    The SimplePCAPynPoint is a wrapper around the simple full frame PCA
    implemented in `PynPoint <https://pynpoint.readthedocs.io/en/latest/\
    pynpoint.util.html#pynpoint.util.psf.pca_psf_subtraction>`__.
    It only allows to compute the residuals for a fixed number of PCA
    components. The advantage of this wrapper over
    :meth:`~MultiComponentPCAPynPoint` is, that it does not require to
    create a PynPoint database file.
    """

    def __init__(
            self,
            num_pca: int):
        """
        Constructor of the class.

        Args:
            num_pca: The number of PCA components to be used.

        """
        self.num_pca = num_pca

    def get_method_keys(self) -> List[str]:
        """
        Get the method name "PCA (#num_pca components)".

        Returns:
            A list with one string "PCA (#num_pca components)".
        """

        return ["PCA (" + str(self.num_pca).zfill(3) + " components)", ]

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Compute the full-frame PCA.

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
            A dictionary which contains the residual of the PCA reduction with
            the dict-key "PCA (#num_pca components)".
        """

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
    """
    The MultiComponentPCAPynPoint is a wrapper around the full
    frame PCA implemented in `PynPoint <https://pynpoint.readthedocs.io/en/\
    latest/pynpoint.processing.html#pynpoint.processing.psfsubtraction.\
    PcaPsfSubtractionModule>`__. While the wrapper
    :meth:`~SimplePCAPynPoint` only accepts a single fixed number
    of PCA components, MultiComponentPCAPynPoint computes several residuals
    with different number of components. This can be more efficient as the
    PCA basis needs to be computed only once. The disadvantage over
    :meth:`~SimplePCAPynPoint` is that MultiComponentPCAPynPoint needs to
    create a Pynpoint database and delete it after computing the residuals.
    """

    def __init__(
            self,
            num_pcas: List[int],
            scratch_dir: Path,
            num_cpus_pynpoint: int = 1):
        """
        Constructor of the class.

        Args:
            num_pcas: List of the number of PCA components to be used.
            scratch_dir: A directory to store the Pynpoint database. Any
                Pynpoint database created during the computation will be deleted
                afterwards.
            num_cpus_pynpoint: Number of CPU cores used by Pynpoint.
        """

        self.num_pcas = num_pcas
        self.scratch_dir = scratch_dir
        self.num_cpus_pynpoint = num_cpus_pynpoint

    def get_method_keys(self) -> List[str]:
        """
        Get the method name "PCA (#num_pca components)".

        Returns:
            A list with strings "PCA (#num_pca components)" (one for each
            value in num_pcas.

        """
        keys = ["PCA (" + str(num_pcas).zfill(3) + " components)"
                for num_pcas in self.num_pcas]

        return keys

    def __call__(
            self,
            stack_with_fake_planet: np.ndarray,
            parang_rad: np.ndarray,
            psf_template: np.ndarray,
            exp_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Compute the full-frame PCA for several numbers of PCA components.

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
            A dictionary which contains the residuals of the PCA reduction with
            the dict-keys "PCA (#num_pca components)".
        """

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Disable all print messages from pynpoint
            sys.stdout = open(os.devnull, 'w')

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

            # Enable print messages again
            sys.stdout = sys.__stdout__

        return result_dict
