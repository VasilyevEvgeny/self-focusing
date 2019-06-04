import abc
from pyfftw.builders import ifft2
from numpy import sqrt, pi, exp, zeros, float64, complex64, correlate, var
from numpy import random, mean
from numpy import max as maximum
from numpy.fft import fftshift
from numba import jit


class Noise(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._r_corr_in_meters = kwargs['r_corr_in_meters']
        self._r_corr_in_points = None

        self._n_x, self._n_y, self._dx, self._dy = None, None, None, None
        self._noise_field, self._noise_field_real, self._noise_field_imag = None, None, None

        self._autocorr_real_x, self._autocorr_real_y, self._autocorr_imag_x, self._autocorr_imag_y = \
            None, None, None, None

    @property
    def r_corr_in_meters(self):
        return self._r_corr_in_meters

    @property
    def noise_field(self):
        return self._noise_field

    @property
    def autocorrs(self):
        return self._autocorr_real_x, self._autocorr_real_y, self._autocorr_imag_x, self._autocorr_imag_y

    def initialize(self, **params):
        self._n_x = params['n_x']
        self._n_y = params['n_y']
        self._dx = params['dx']
        self._dy = params['dy']

        self._r_corr_in_points = self._r_corr_in_meters // max(self._dx, self._dy)

    @abc.abstractmethod
    def process(self):
        """Generates gaussian_noise field and autocorrelation functions"""

    @staticmethod
    def calculate_autocorr(noise, n_iter, n, type):
        autocorr = zeros(shape=(n,), dtype=float64)
        for i in range(n_iter):
            if type == 'x':
                autocorr += correlate(noise[i, :], noise[i, :], mode='same')
            elif type == 'y':
                autocorr += correlate(noise[:, i], noise[:, i], mode='same')
            else:
                raise Exception('Wrong type!')
            autocorr /= n_iter

        return autocorr

    def calculate_autocorrelations(self, field_real, field_imag, n_x, n_y):
        autocorr_real_x = self.calculate_autocorr(field_real, n_x, n_y, 'x')
        autocorr_real_y = self.calculate_autocorr(field_real, n_y, n_x, 'y')
        autocorr_imag_x = self.calculate_autocorr(field_imag, n_x, n_y, 'x')
        autocorr_imag_y = self.calculate_autocorr(field_imag, n_y, n_x, 'y')

        return autocorr_real_x, autocorr_real_y, autocorr_imag_x, autocorr_imag_y

    @staticmethod
    def find_r_corr_index(arr):
        n = len(arr)
        th = arr[n // 2] * exp(-1.0)
        for i in range(n // 2, n, 1):
            if arr[i] < th:
                return i - n // 2

    def calculate_r_corr(self):
        r_corr_real_x = self._dx * self.find_r_corr_index(self._autocorr_real_x)
        r_corr_real_y = self._dy * self.find_r_corr_index(self._autocorr_real_y)
        r_corr_imag_x = self._dx * self.find_r_corr_index(self._autocorr_imag_x)
        r_corr_imag_y = self._dy * self.find_r_corr_index(self._autocorr_imag_y)

        return mean([r_corr_real_x, r_corr_real_y, r_corr_imag_x, r_corr_imag_y])


class GaussianNoise(Noise):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__variance_expected = kwargs.get('variance', 1)

    @property
    def variance_expected(self):
        return self.__variance_expected

    @property
    def variance_real(self):
        return var(self._noise_field_real)

    @property
    def variance_imag(self):
        return var(self._noise_field_imag)

    @staticmethod
    @jit(nopython=True)
    def generate_protoarray(n_x, n_y, variance, r_corr_in_points):
        proto = zeros(shape=(n_x, n_y), dtype=complex64)

        scale = r_corr_in_points / max(n_x, n_y)
        cf = scale * sqrt(pi * variance)
        d = 0.5 * (pi * scale) ** 2
        amplitude = sqrt(12)

        for i in range(n_x):
            for j in range(n_y):
                a, b = amplitude * (random.random()-0.5), amplitude * (random.random()-0.5)
                gauss = cf * exp(-d * ((i-n_x//2) ** 2 + (j-n_y//2) ** 2))
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

        for i in range(n_x):
            for j in range(n_y):
                real_part[i, j] = noise_field[i, j].real
                imag_part[i, j] = noise_field[i, j].imag

        return real_part, imag_part

    def process(self):
        proto = self.generate_protoarray(self._n_x, self._n_y, self.__variance_expected, self._r_corr_in_points)
        proto_shifted = fftshift(proto, axes=(0, 1))
        proto_fft_obj = ifft2(proto_shifted)
        proto_fft_normalized = self.normalize_after_fft(proto_fft_obj())

        self._noise_field = proto_fft_normalized
        self._noise_field_real, self._noise_field_imag = \
            self.initialize_noise_arrays(self._noise_field, self._n_x, self._n_y)

        self._autocorr_real_x, self._autocorr_real_y, self._autocorr_imag_x, self._autocorr_imag_y = \
            self.calculate_autocorrelations(self._noise_field_real, self._noise_field_imag, self._n_x, self._n_y)
