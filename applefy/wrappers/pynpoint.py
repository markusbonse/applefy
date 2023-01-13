import numpy as np
from pynpoint.util.psf import pca_psf_subtraction

from applefy.detections.contrast import DataReductionInterface


class SimplePynPointPCA(DataReductionInterface):

    def __init__(self, num_pca):
        self.num_pca = num_pca

    def get_method_keys(self):
        return ["PCA (" + str(self.num_pca) + " components)", ]

    def __call__(
            self,
            stack_with_fake_planet,
            parang_rad):

        _, residual_stack_fake = pca_psf_subtraction(
            images=stack_with_fake_planet,
            angles=-np.rad2deg(parang_rad),
            pca_number=self.num_pca)

        # Average along temporal axis
        residual_image = np.mean(residual_stack_fake, axis=0)

        result_dict = dict()
        result_dict["PCA (" + str(self.num_pca) + " components)"] = \
            residual_image

        return result_dict
