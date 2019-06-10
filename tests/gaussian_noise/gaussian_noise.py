from unittest import TestCase
from numpy import mean
from numpy.random import random, randint
from tqdm import trange

from core import GaussianNoise


class TestGaussianNoise(TestCase):
    """
    Class for testing of generated complex gaussian noise parameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__n_x = 512
        self.__n_y = 511
        self.__x_max = 500 * 10**-6
        self.__y_max = 500 * 10 ** -6
        self.__dx = self.__x_max / self.__n_x
        self.__dy = self.__y_max / self.__n_y

        self.__eps_var = 0.2
        self.__eps_r_corr = 0.6

    def test_gaussian_noise(self, n_epochs=50):

        for _ in trange(n_epochs, desc='gaussian_noise'):
            variance_expected = 0.5 + random()
            r_corr_expected = randint(5, 15) * 10 ** -6

            gaussian_noise = GaussianNoise(r_corr_in_meters=r_corr_expected,
                                           variance=variance_expected)

            gaussian_noise.initialize(n_x=self.__n_x,
                                      n_y=self.__n_y,
                                      dx=self.__dx,
                                      dy=self.__dy)
            gaussian_noise.process()

            variance_generated = mean([gaussian_noise.variance_real, gaussian_noise.variance_imag])
            self.assertLess(abs(variance_expected - variance_generated) / variance_expected, self.__eps_var)

            r_corr_generated = gaussian_noise.calculate_r_corr()
            self.assertLess(abs(r_corr_expected - r_corr_generated) / r_corr_expected, self.__eps_r_corr)

            del gaussian_noise
