from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from numpy import exp, conj, zeros, complex64
from numba import jit
from pyfftw.builders import fft2, ifft2


class DiffractionExecutor(metaclass=ABCMeta):
    """
    Abstract class for diffraction object.
    The class takes on the input in the constructor a beam object, which contains all the necessary beam parameters
    for further calculations.

    Depending on the type of beam diffraction is implemented in different ways. To simulate diffraction in
    a 2-dimensional beam and a 3-dimensional beam in the axisymmetric approximation, a sweep was used;
    to simulate the diffraction of a non-axisymmetric 3-dimensional beam, we used fast Fourier transform with pyfftw
    in several threads.

    """
    def __init__(self, **kwargs):
        self._beam = kwargs['beam']

    @abstractmethod
    def info(self):
        """DiffractionExecutor type"""

    @abstractmethod
    def process_diffraction(self, dz):
        """Process_diffraction"""


class SweepDiffractionExecutorX(DiffractionExecutor):
    """
    Class for modeling the diffraction of a 2-dimensional beam using sweep.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # sweep coefficients and arrays

        self.__c1 = 1.0 / (2.0 * self._beam.dx ** 2)
        self.__c2 = 2j * self._beam.medium.k_0

        self.__alpha = self.__c1
        self.__gamma = self.__c1

        self.__kappa_left, self.__mu_left, self.__kappa_right, self.__mu_right = \
            0.0, 0.0, 0.0, 0.0

        self.__delta = zeros(shape=(self._beam.n_x,), dtype=complex64)
        self.__xi = zeros(shape=(self._beam.n_x,), dtype=complex64)
        self.__eta = zeros(shape=(self._beam.n_x,), dtype=complex64)

    @property
    def info(self):
        return 'sweep_diffraction_executor_x'

    @staticmethod
    @jit(nopython=True)
    def __fast_process(field, n_x, dz, c1, c2, alpha, gamma, delta, xi, eta,
                       kappa_left, mu_left, kappa_right, mu_right):

        # left boundary condition
        xi[1], eta[1] = kappa_left, mu_left

        # forward
        for i in range(1, n_x - 1):
            beta = 2.0 * c1 + c2 / dz
            delta[i] = alpha * field[i + 1] - \
                       conj(beta) * field[i] + \
                       gamma * field[i - 1]
            xi[i + 1] = alpha / (beta - gamma * xi[i])
            eta[i + 1] = (delta[i] + gamma * eta[i]) / \
                         (beta - gamma * xi[i])

        # right boundary condition
        field[n_x - 1] = (mu_right + kappa_right * eta[n_x - 1]) / \
                         (1.0 - kappa_right * xi[n_x - 1])

        # backward
        for j in range(n_x - 1, 0, -1):
            field[j - 1] = xi[j] * field[j] + eta[j]

        return field

    def process_diffraction(self, dz):
        """
        :param dz: current step along evolutionary coordinate z

        :return: None
        """
        self._beam._field = self.__fast_process(self._beam._field, self._beam.n_x, dz, self.__c1,
                                                self.__c2, self.__alpha, self.__gamma, self.__delta, self.__xi,
                                                self.__eta, self.__kappa_left, self.__mu_left, self.__kappa_right,
                                                self.__mu_right)


class SweepDiffractionExecutorR(DiffractionExecutor):
    """
    Class for modeling the diffraction of a 3-dimensional beam in axisymmetric approximation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # sweep coefficients and arrays

        self.__c1 = 1.0 / (2.0 * self._beam.dr ** 2)
        self.__c2 = 1.0 / (4.0 * self._beam.dr)
        self.__c3 = 2j * self._beam.medium.k_0

        self.__alpha = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__beta = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__gamma = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__vx = zeros(shape=(self._beam.n_r,), dtype=complex64)  # array responsible for accounting topological
                                                                     # charge

        for i in range(1, self._beam.n_r - 1):
            self.__alpha[i] = self.__c1 + self.__c2 / self._beam.rs[i]
            self.__gamma[i] = self.__c1 - self.__c2 / self._beam.rs[i]
            self.__vx[i] = (self._beam.m / self._beam.rs[i]) ** 2  # topological charge accounting

        self.__kappa_left, self.__mu_left, self.__kappa_right, self.__mu_right = \
            1.0, 0.0, 0.0, 0.0

        self.__delta = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__xi = zeros(shape=(self._beam.n_r,), dtype=complex64)
        self.__eta = zeros(shape=(self._beam.n_r,), dtype=complex64)

    @property
    def info(self):
        return 'sweep_diffraction_executor_r'

    @staticmethod
    @jit(nopython=True)
    def __fast_process(field, n_r, dz, c1, c3, alpha, beta, gamma, delta, xi, eta, vx,
                     kappa_left, mu_left, kappa_right, mu_right):

        # left boundary condition
        xi[1], eta[1] = kappa_left, mu_left

        # forward
        for i in range(1, n_r - 1):
            beta[i] = 2.0 * c1 + c3 / dz + vx[i]
            delta[i] = alpha[i] * field[i + 1] - \
                       (conj(beta[i]) - vx[i]) * field[i] + \
                       gamma[i] * field[i - 1]
            xi[i + 1] = alpha[i] / (beta[i] - gamma[i] * xi[i])
            eta[i + 1] = (delta[i] + gamma[i] * eta[i]) / \
                         (beta[i] - gamma[i] * xi[i])

        # right boundary condition
        field[n_r - 1] = (mu_right + kappa_right * eta[n_r - 1]) / \
                         (1.0 - kappa_right * xi[n_r - 1])

        # backward
        for j in range(n_r - 1, 0, -1):
            field[j - 1] = xi[j] * field[j] + eta[j]

        return field

    def process_diffraction(self, dz):
        """
        :param dz: current step along evolutionary coordinate z

        :return: None
        """
        self._beam._field = self.__fast_process(self._beam._field, self._beam.n_r, dz, self.__c1,
                                                self.__c3, self.__alpha, self.__beta, self.__gamma, self.__delta,
                                                self.__xi, self.__eta, self.__vx, self.__kappa_left, self.__mu_left,
                                                self.__kappa_right, self.__mu_right)


class FourierDiffractionExecutorXY(DiffractionExecutor):
    """
    Class for modeling the diffraction of a 3-dimensional beam using fast Fourier transform in pyfftw.
    """

    MAX_NUMBER_OF_CPUS = cpu_count()  # number of threads for parallelization

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'fourier_diffraction_executor_xy'

    @staticmethod
    @jit(nopython=True)
    def __phase_increment(field_fft, n_x, n_y, k_xs, k_ys, current_lin_phase):
        """
        :param field_fft: spatial spectrum of the field array
        :param n_x: number of points in spatial grid along x
        :param n_y: number of points in spatial grid along y
        :param k_xs: wave vector grid nosed along x
        :param k_ys: wave vector grid nosed along y
        :param current_lin_phase: linear phase shift in spectral space

        :return: spatial spectrum of the field array with linear phase shift along both axes
        """
        for i in range(n_x):
            field_fft[i, :] *= exp(current_lin_phase * k_xs[i] ** 2)

        for j in range(n_y):
            field_fft[:, j] *= exp(current_lin_phase * k_ys[j] ** 2)

        return field_fft

    def process_diffraction(self, dz, n_jobs=MAX_NUMBER_OF_CPUS):
        """
        :param dz: current step along evolutionary coordinate z
        :param n_jobs: number of threads for parallelization

        :return: None
        """

        # calculation of current linear phase shift
        current_lin_phase = 0.5j * dz / self._beam.medium.k_0

        # forward parallel fast Fourier transform
        fft_obj = fft2(self._beam._field, threads=n_jobs)

        # linear phase increment
        field_fft = self.__phase_increment(fft_obj(), self._beam.n_x, self._beam.n_y, self._beam.k_xs,
                                           self._beam.k_ys, current_lin_phase)

        # backward parallel fast Fourier transform
        ifft_obj = ifft2(field_fft, threads=n_jobs)

        # field initialization with updated values
        self._beam._field = ifft_obj()
