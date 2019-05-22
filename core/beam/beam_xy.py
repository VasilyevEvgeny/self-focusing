from numpy import pi, conj, arctan2, exp, sqrt, zeros, float64, complex64, mean
from numpy import max as maximum
from scipy.special import gamma
from numba import jit

from .beam_3d import Beam3D


class BeamXY(Beam3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__x_0 = kwargs['x_0']
        self.__y_0 = kwargs['y_0']

        self.__x_max = 10.0 * max(self.__x_0, self.__y_0)
        self.__y_max = self.__x_max

        self.__n_x = kwargs['n_x']
        self.__n_y = kwargs['n_y']

        self.__dx = self.__x_max / self.__n_x
        self.__dy = self.__y_max / self.__n_y

        self.__xs = [i * self.__dx - 0.5 * self.__x_max for i in range(self.__n_x)]
        self.__ys = [i * self.__dy - 0.5 * self.__y_max for i in range(self.__n_y)]

        self.__dk_x = 2.0 * pi / self.__x_max
        self.__dk_y = 2.0 * pi / self.__y_max

        self.__k_xs = [i * self.__dk_x if i < self.__n_x / 2 else (i - self.__n_x) * self.__dk_x for i in range(self.__n_x)]
        self.__k_ys = [i * self.__dk_y if i < self.__n_y / 2 else (i - self.__n_y) * self.__dk_y for i in range(self.__n_y)]

        self.__noise_percent = kwargs['noise_percent']
        self.__noise = kwargs['noise']
        self.__noise.initialize(n_x=self.__n_x,
                                n_y=self.__n_y,
                                dx=self.__dx,
                                dy=self.__dy)
        self.__noise.process()

        self._field = self.initialize_field(self._M, self._m, self.__x_0, self.__y_0, self.__x_max, self.__y_max,
                                            self.__dx, self.__dy, self.__n_x, self.__n_y, self.__noise_percent,
                                            self.__noise.noise_field)

        self._i_0 = self.calculate_i0()
        self._z_diff = self._medium.k_0 * mean([self.__x_0, self.__y_0])**2

        self._r_kerr = 2 * self.medium.k_0 * self.medium.n_2 * self._i_0 * self._z_diff / self.medium.n_0

        self.update_intensity()

    @property
    def info(self):
        return 'beam_xy'

    @property
    def x_0(self):
        return self.__x_0

    @property
    def y_0(self):
        return self.__x_0

    @property
    def x_max(self):
        return self.__x_max

    @property
    def y_max(self):
        return self.__y_max

    @property
    def n_x(self):
        return self.__n_x

    @property
    def n_y(self):
        return self.__n_y

    @property
    def dx(self):
        return self.__dx

    @property
    def dy(self):
        return self.__dy

    @property
    def xs(self):
        return self.__xs

    @property
    def ys(self):
        return self.__ys

    @property
    def k_xs(self):
        return self.__k_xs

    @property
    def k_ys(self):
        return self.__k_ys

    @property
    def noise_percent(self):
        return self.__noise_percent

    @property
    def i_0(self):
        return self._i_0

    @property
    def z_diff(self):
        return self._z_diff

    @property
    def noise(self):
        return self.__noise

    def update_intensity(self):
        self._intensity = self.field_to_intensity(self._field, self.__n_x, self.__n_y)
        self._i_max = maximum(self._intensity)

    @staticmethod
    @jit(nopython=True)
    def field_to_intensity(field, n_x, n_y):
        intensity = zeros(shape=(n_x, n_y), dtype=float64)
        for i in range(n_x):
            for j in range(n_y):
                intensity[i, j] = (field[i, j] * conj(field[i, j])).real

        return intensity

    @staticmethod
    @jit(nopython=True)
    def calculate_intensity_intergral(field, n_x, n_y, dx, dy):
        intensity_intergral = 0.0
        for i in range(n_x):
            for j in range(n_y):
                intensity_intergral += (field[i, j] * conj(field[i, j])).real * dx * dy

        return intensity_intergral

    def calculate_i0(self):
        if self.__noise_percent == 0.0 and self.__x_0 == self.__y_0:
            return self._p_0 / (pi * self.__x_0**2 * gamma(self._m+1))
        else:
            return self._p_0 / self.calculate_intensity_intergral(self._field, self.__n_x, self.__n_y, self.__dx, self.__dy)

    @staticmethod
    @jit(nopython=False)
    def initialize_field(M, m, x_0, y_0, x_max, y_max, dx, dy, n_x, n_y, noise_percent, noise):
        arr = zeros(shape=(n_x, n_y), dtype=complex64)
        for i in range(n_x):
            for j in range(n_y):
                x, y = i * dx - 0.5 * x_max, j * dy - 0.5 * y_max
                arr[i, j] = (1.0 + 0.01 * noise_percent * noise[i, j]) * \
                            sqrt((x / x_0)**2 + (y / y_0)**2)**M * \
                            exp(-0.5 * ((x / x_0) ** 2 + (y / y_0) ** 2)) * \
                            exp(1j * m * arctan2(x, y))

        return arr
