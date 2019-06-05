from numpy import pi, conj, exp, zeros, float64, complex64
from numpy import max as maximum
from scipy.special import gamma
from numba import jit

from .beam_3d import Beam3D


class BeamR(Beam3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__r_0 = kwargs['r_0']
        self.__r_max = 40.0 * self.__r_0
        self.__n_r = kwargs['n_r']
        self.__dr = self.__r_max / self.__n_r
        self.__rs = [i * self.__dr for i in range(self.__n_r)]

        self._field = self.__initialize_field(self._M, self.__r_0, self.__dr, self.__n_r)

        self._i_0 = self.__calculate_i0()
        self._z_diff = self._medium.k_0 * self.__r_0**2

        self._r_kerr = 2 * self.medium.k_0 * self.medium.n_2 * self._i_0 * self._z_diff / self.medium.n_0

        self.update_intensity()

    @property
    def info(self):
        return 'beam_r'

    @property
    def r_0(self):
        return self.__r_0

    @property
    def r_max(self):
        return self.__r_max

    @property
    def n_r(self):
        return self.__n_r

    @property
    def rs(self):
        return self.__rs

    @property
    def dr(self):
        return self.__dr

    def update_intensity(self):
        self._intensity = self.__field_to_intensity(self._field, self.__n_r)
        self._i_max = maximum(self._intensity)

    @staticmethod
    @jit(nopython=True)
    def __field_to_intensity(field, n_r):
        intensity = zeros(shape=(n_r,), dtype=float64)
        for i in range(n_r):
            intensity[i] = (field[i] * conj(field[i])).real

        return intensity

    def __calculate_i0(self):
        return self._p_0 / (pi * self.__r_0**2 * gamma(self._m+1))

    @staticmethod
    @jit(nopython=True)
    def __initialize_field(M, r_0, dr, n_r):
        arr = zeros(shape=(n_r,), dtype=complex64)
        for i in range(n_r):
            r = i * dr
            arr[i] = (r / r_0)**M * exp(-0.5 * (r / r_0)**2)

        return arr
