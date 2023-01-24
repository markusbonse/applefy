import numpy as np
from vip_hci.psfsub.pca_fullfr import pca

from applefy.detections.contrast import DataReductionInterface


class SimplePCAvip(DataReductionInterface):

    def __init__(self,
                 num_pca,
                 kwarg=None):
        self.num_pca = num_pca

    def get_method_keys(self):
        return ["PCA (" + str(self.num_pca).zfill(3) + " components)", ]

    def __call__(
            self,
            stack_with_fake_planet,
            parang_rad,
            psf_template,
            exp_id):

        residual = pca(
            cube=stack_with_fake_planet,
            angle_list=np.rad2deg(parang_rad),
            ncomp=self.num_pca)

        # save the result
        result_dict = dict()
        result_dict["PCA (" + str(self.num_pca).zfill(3) + " components)"] = \
            residual

        return result_dict


class MultiComponentPCAvip(DataReductionInterface):

    def __init__(
            self,
            num_pcas_tuple,
            kwarg=None):

        self.num_pcas_tuple = num_pcas_tuple
        self.num_pcas = np.arange(
            self.num_pcas_tuple[0],
            self.num_pcas_tuple[1] + self.num_pcas_tuple[2],
            self.num_pcas_tuple[2])

        if kwarg is None:
            self.kwarg = dict()
        else:
            self.kwarg = kwarg

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
