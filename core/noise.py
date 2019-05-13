import abc
from pyfftw.builders import ifft2
from numpy import sqrt, pi, exp, zeros, float64, complex64, correlate, var
from numpy import random
from numpy.fft import fftshift
from numba import jit


class Noise(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._r_corr_in_meters = kwargs['r_corr_in_meters']
        self._r_corr_in_points = None

        self._n_x, self._n_y, self._dx, self._dy = None, None, None, None
        self._noise_field, self._noise_field_real, self._noise_field_imag, self._noise_field_norm, self._noise_field_summ = \
            None, None, None, None, None

        self._autocorr_x, self._autocorr_y = None, None

    @property
    def r_corr_in_meters(self):
        return self._r_corr_in_meters

    @property
    def noise_field_summ(self):
        return self._noise_field_summ

    @property
    def noise_field_norm(self):
        return self._noise_field_norm

    @property
    def autocorrs(self):
        return self._autocorr_x, self._autocorr_y

    def initialize(self, **params):
        self._n_x = params['n_x']
        self._n_y = params['n_y']
        self._dx = params['dx']
        self._dy = params['dy']

        self._r_corr_in_points = self._r_corr_in_meters // max(self._dx, self._dy)

    @abc.abstractmethod
    def process(self):
        """Generates noise field and autocorrelation functions"""

    @staticmethod
    def calculate_autocorr(noise, n_iter, n, type):
        autocorr = zeros(shape=(2 * n - 1,), dtype=float64)
        for i in range(n_iter):
            if type == 'x':
                autocorr += correlate(noise[:, i], noise[:, i], mode='full')
            elif type == 'y':
                autocorr += correlate(noise[i, :], noise[i, :], mode='full')
            else:
                raise Exception('Wrong type!')
            autocorr /= n_iter

        return autocorr

    def calculate_autocorrelations(self, noise_field_summ, n_x, n_y):
        autocorr_x = self.calculate_autocorr(noise_field_summ, n_y, n_x, 'x')
        autocorr_y = self.calculate_autocorr(noise_field_summ, n_x, n_y, 'y')

        return autocorr_x, autocorr_y


class GaussianNoise(Noise):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__variance_expected = kwargs.get('variance', 1)

    @property
    def variance_expected(self):
        return self.__variance_expected

    @property
    def variance_real(self):
        return var(self._noise_field_summ)

    @staticmethod
    @jit(nopython=True)
    def generate_protoarray(n_x, n_y, variance, r_corr_in_points):
        n_rows = n_x
        n_cols = n_y
        proto = zeros(shape=(n_rows, n_cols), dtype=complex64)

        scale = r_corr_in_points / max(n_rows, n_cols)
        cf = scale * sqrt(pi * variance)
        d = 0.5 * (pi * scale) ** 2
        amplitude = sqrt(22)

        for i in range(n_rows):
            for j in range(n_cols):
                a, b = amplitude * (random.random() - 0.5), amplitude * (random.random() - 0.5)
                gauss = cf * exp(-d * (i ** 2 + j ** 2))
                proto[i, j] = a * gauss + 1j * b * gauss

        return proto

    @staticmethod
    @jit(nopython=True)
    def normalize_after_fft(arr):
        n1, n2 = arr.shape[0], arr.shape[1]
        for i in range(n1):
            for j in range(n2):
                arr[i, j] *= n1 * n2

        return arr

    @staticmethod
    def initialize_noise_arrays(noise_field, n_x, n_y):
        real_part = zeros(shape=(n_x, n_y), dtype=float64)
        imag_part = zeros(shape=(n_x, n_y), dtype=float64)
        norm = zeros(shape=(n_x, n_y), dtype=float64)
        summ = zeros(shape=(n_x, n_y), dtype=float64)

        for i in range(n_x):
            for j in range(n_y):
                real_part[i, j] = noise_field[i, j].real
                imag_part[i, j] = noise_field[i, j].imag
                norm[i, j] = (noise_field[i, j] * noise_field[i, j].conjugate()).real
                summ[i, j] = noise_field[i, j].real + noise_field[i, j].imag

        return real_part, imag_part, norm, summ

    def process(self):
        proto = self.generate_protoarray(self._n_x, self._n_y, self.__variance_expected, self._r_corr_in_points)
        proto_shifted = fftshift(proto, axes=(0, 1))
        proto_fft_obj = ifft2(proto_shifted)
        proto_fft_normalized = self.normalize_after_fft(proto_fft_obj())

        self._noise_field = proto_fft_normalized
        self._noise_field_real, self._noise_field_imag, self._noise_field_norm, self._noise_field_summ = \
            self.initialize_noise_arrays(self._noise_field, self._n_x, self._n_y)

        self._autocorr_x, self._autocorr_y = \
            self.calculate_autocorrelations(self._noise_field_summ, self._n_x, self._n_y)
