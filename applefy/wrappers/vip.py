"""
The following wrapper classes allow to use
`VIP <https://vip.readthedocs.io/en/latest/>`__ with applefy.
VIP is not on the requirement list of applefy. It has to be installed
separately.
"""

from typing import List, Dict, Optional, Tuple

import numpy as np
from vip_hci.psfsub.pca_fullfr import pca

from applefy.detections.contrast import DataReductionInterface


class SimplePCAvip(DataReductionInterface):
    """
    The SimplePCAvip is a wrapper around the simple full frame PCA
    implemented in `VIP <https://vip.readthedocs.io/en/latest/vip_hci.\
    psfsub.html#vip_hci.psfsub.pca_fullfr.pca>`__.
    It only allows to compute the residuals for a fixed number of PCA
    components.
    """

    def __init__(
            self,
            num_pca: int,
            kwarg: Optional[dict] = None):
        """
        Constructor of the class.

        Args:
            num_pca: The number of PCA components to be used.
            kwarg: Additional arguments (see documentation in VIP).

        """
        self.num_pca = num_pca

        if kwarg is None:
            self.kwarg = dict()
        else:
            self.kwarg = kwarg

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


        residual = pca(
            cube=stack_with_fake_planet,
            angle_list=np.rad2deg(parang_rad),
            ncomp=self.num_pca,
            **self.kwarg)

        # save the result
        result_dict = dict()
        result_dict["PCA (" + str(self.num_pca).zfill(3) + " components)"] = \
            residual

        return result_dict


class MultiComponentPCAvip(DataReductionInterface):
    """
    The MultiComponentPCAvip is a wrapper around the full
    frame PCA implemented in `VIP <https://vip.readthedocs.io/en/latest/\
    vip_hci.psfsub.html#vip_hci.psfsub.pca_fullfr.pca>`__. While the wrapper
    :meth:`~SimplePCAvip` only accepts a single fixed number
    of PCA components, MultiComponentPCAvip computes several residuals
    with different number of components.
    """

    def __init__(
            self,
            num_pcas_tuple: Tuple[int, int, int],
            kwarg: Optional[dict] = None):
        """
        Constructor of the class.

        Args:
            num_pcas_tuple: A tuple which defines the range of components to be
                computed: (Min components, Max components, steps).
            kwarg: Additional arguments (see documentation in VIP).
        """

        self.num_pcas_tuple = num_pcas_tuple
        self.num_pcas = np.arange(
            self.num_pcas_tuple[0],
            self.num_pcas_tuple[1] + self.num_pcas_tuple[2],
            self.num_pcas_tuple[2])

        if kwarg is None:
            self.kwarg = dict()
        else:
            self.kwarg = kwarg

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

        residuals = pca(
            cube=stack_with_fake_planet,
            angle_list=np.rad2deg(parang_rad),
            ncomp=self.num_pcas_tuple,
            **self.kwarg)

        # bring the results into the required shape
        result_dict = dict()

        for idx, tmp_algo_name in enumerate(self.get_method_keys()):
            result_dict[tmp_algo_name] = residuals[idx]

        return result_dict
