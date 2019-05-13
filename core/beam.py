import abc
from numpy import pi, conj, arctan2, exp, sqrt, zeros, float64, complex64, mean
from numpy import max as maximum
from scipy.special import gamma
from numba import jit

from .medium import Medium
from .m_constants import M_Constants


class Beam(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.m_constants = M_Constants()
        self.lmbda = kwargs['lmbda']

        self.medium = Medium(name=kwargs['medium'],
                             lmbda=self.lmbda,
                             m_constants=self.m_constants)

        self.Pcr_G = self.calculate_Pcr_G()

        self.M = kwargs['M']
        self.m = kwargs['m']
        if self.m == 0:
            if self.M == 0:
                self.distribution_type = 'gauss'
            else:
                self.distribution_type = 'ring'
        else:
            if self.M == 0:
                raise Exception('Gauss with vortex phase is a wrong initial mode!')
            self.distribution_type = 'vortex'

        if self.distribution_type == 'gauss':
            self.P0_to_Pcr_G = kwargs['P0_to_Pcr_G']
            self.p_0 = self.P0_to_Pcr_G * self.Pcr_G
        elif self.distribution_type == 'ring':
            self.P0_to_Pcr_G = kwargs['P0_to_Pcr_G']
            self.p_0 = self.P0_to_Pcr_G * self.Pcr_G
        elif self.distribution_type == 'vortex':
            self.Pcr_V = self.calculate_Pcr_V()
            self.P0_to_Pcr_V = kwargs['P0_to_Pcr_V']
            self.p_0 = self.P0_to_Pcr_V * self.Pcr_V
        else:
            raise Exception('Wrong distribution type: "%s".' % self.distribution_type)

        self.intensity, self.i_max = None, None

    @abc.abstractmethod
    def info(self):
        """Information about beam"""

    def calculate_Pcr_G(self):
        return 3.77 * self.lmbda ** 2 / (8 * pi * self.medium.n_0 * self.medium.n_2)

    def calculate_Pcr_V(self):
        return self.Pcr_G * 2**(2 * self.m + 1) * gamma(self.m + 1) * gamma(self.m + 2) / (2 * gamma(2 * self.m + 1))


class Beam_R(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.r_0 = kwargs['r_0']
        self.r_max = 40.0 * self.r_0
        self.n_r = kwargs['n_r']
        self.dr = self.r_max / self.n_r
        self.rs = [i * self.dr for i in range(self.n_r)]
        self.dk_r = 2.0 * pi / self.r_max
        self.k_rs = [i * self.dk_r for i in range(self.n_r)]

        self.field = self.initialize_field(self.M, self.r_0, self.dr, self.n_r)

        self.i_0 = self.calculate_i0()
        self.z_diff = self.medium.k_0 * self.r_0**2

        self.update_intensity()

    @property
    def info(self):
        return 'beam_r'

    def update_intensity(self):
        self.intensity = self.field_to_intensity(self.field, self.n_r)
        self.i_max = maximum(self.intensity)

    @staticmethod
    @jit(nopython=True)
    def field_to_intensity(field, n_r):
        intensity = zeros(shape=(n_r,), dtype=float64)
        for i in range(n_r):
            intensity[i] = (field[i] * conj(field[i])).real

        return intensity

    def calculate_i0(self):
        return self.p_0 / (pi * self.r_0**2 * gamma(self.m+1))

    @staticmethod
    @jit(nopython=True)
    def initialize_field(M, r_0, dr, n_r):
        arr = zeros(shape=(n_r,), dtype=complex64)
        for i in range(n_r):
            r = i * dr
            arr[i] = (r / r_0)**M * exp(-0.5 * (r / r_0)**2)

        return arr


class Beam_XY(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x_0 = kwargs['x_0']
        self.y_0 = kwargs['y_0']

        self.x_max = 10.0 * max(self.x_0, self.y_0)
        self.y_max = self.x_max

        self.n_x = kwargs['n_x']
        self.n_y = kwargs['n_y']

        self.dx = self.x_max / self.n_x
        self.dy = self.y_max / self.n_y

        self.xs = [i * self.dx - 0.5 * self.x_max for i in range(self.n_x)]
        self.ys = [i * self.dy - 0.5 * self.y_max for i in range(self.n_y)]

        self.dk_x = 2.0 * pi / self.x_max
        self.dk_y = 2.0 * pi / self.y_max

        self.k_xs = [i * self.dk_x if i < self.n_x / 2 else (i - self.n_x) * self.dk_x for i in range(self.n_x)]
        self.k_ys = [i * self.dk_y if i < self.n_y / 2 else (i - self.n_y) * self.dk_y for i in range(self.n_y)]

        self.noise_percent = kwargs['noise_percent']
        self.noise = kwargs['noise']
        self.noise.initialize(n_x=self.n_x,
                              n_y=self.n_y,
                              dx=self.dx,
                              dy=self.dy)
        self.noise.process()

        self.field = self.initialize_field(self.M, self.m, self.x_0, self.y_0, self.x_max, self.y_max, self.dx, self.dy,
                                           self.n_x, self.n_y, self.noise_percent, self.noise.noise_field_norm)

        self.i_0 = self.calculate_i0()
        self.z_diff = self.medium.k_0 * mean([self.x_0, self.y_0])**2

        self.update_intensity()

    @property
    def info(self):
        return 'beam_xy'

    def update_intensity(self):
        self.intensity = self.field_to_intensity(self.field, self.n_x, self.n_y)
        self.i_max = maximum(self.intensity)

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
        if self.noise_percent == 0.0 and self.x_0 == self.y_0:
            return self.p_0 / (pi * self.x_0**2 * gamma(self.m+1))
        else:
            return self.p_0 / self.calculate_intensity_intergral(self.field, self.n_x, self.n_y, self.dx, self.dy)

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
