from numpy import max as maximum
from numpy import exp, zeros, float64, complex64, conj, heaviside
from numba import jit

from .beam_2d import Beam2D


class BeamX(Beam2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__x_0 = kwargs['x_0']
        self.__x_max = 40.0 * self.__x_0
        self.__n_x = kwargs['n_x']
        self.__dx = self.__x_max / self.__n_x
        self.__xs = [i * self.__dx - 0.5 * self.__x_max for i in range(self.__n_x)]

        if self._M == 0:
            self._distribution_type = 'gauss'
            self.__half = False
        elif self._M > 0:
            self.__half = kwargs['half']
            if self.__half:
                self._distribution_type = 'half of ring'
            else:
                self._distribution_type = 'ring'
        else:
            Exception('Wrong M!')
        self._field = self.initialize_field(self.__half, self._M, self.__x_0, self.__x_max, self.__dx, self.__n_x)

        self._z_diff = self.medium.k_0 * self.__x_0 ** 2

        self._r_kerr = kwargs['r_kerr']
        self._i_0 = 0.5 * self._r_kerr * self.medium.n_0 / (self.medium.k_0 * self.medium.n_2 * self._z_diff)

        self.update_intensity()

    @property
    def info(self):
        return 'beam_x'

    @property
    def x_0(self):
        return self.__x_0

    @property
    def x_max(self):
        return self.__x_max

    @property
    def n_x(self):
        return self.__n_x

    @property
    def xs(self):
        return self.__xs

    @property
    def dx(self):
        return self.__dx

    def update_intensity(self):
        self._intensity = self.field_to_intensity(self._field, self.__n_x)
        self._i_max = maximum(self._intensity)

    @staticmethod
    @jit(nopython=True)
    def field_to_intensity(field, n_x):
        intensity = zeros(shape=(n_x,), dtype=float64)
        for i in range(n_x):
            intensity[i] = (field[i] * conj(field[i])).real

        return intensity

    @staticmethod
    def initialize_field(half, M, x_0, x_max, dx, n_x):
        arr = zeros(shape=(n_x,), dtype=complex64)
        for i in range(n_x):
            x = i * dx - 0.5 * x_max
            arr[i] = ((heaviside(-x, 0) * (1 - half) + heaviside(x, 0)) * (x / x_0)) ** M * exp(-0.5 * (x / x_0) ** 2)

        return arr
