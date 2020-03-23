from numpy import pi, exp, zeros, complex64, save
from scipy.special import gamma
from numba import jit

from .beam_3d import Beam3D
from ..functions import r_to_xy_complex


class BeamR(Beam3D):
    """
    Subsubclass for 3-dimensional beam in axisymmetric approximation with radial coordinate r
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__r_0 = kwargs['r_0']  # characteristic spatial size, [m]
        self.__r_max = self._radii_in_grid * self.__r_0  # spatial grid size, [m]
        self.__n_r = kwargs['n_r']  # number of points in spatial grid
        self.__dr = self.__r_max / self.__n_r  # spatial grid step, [m]
        self.__rs = [i * self.__dr for i in range(self.__n_r)]  # spatial grid nodes, [m]

        # field initialization
        self._field = self.__initialize_field(self._M, self.__r_0, self.__dr, self.__n_r)

        # other parameters initialization
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

    def __calculate_i0(self):
        """
        LATEX SYNTAX:
        P_0 = \int\limits_0^{+\infty} I_0(r) 2 \pi r dr = I_0 \int\limits_0^{+\infty} i(r) 2 \pi r dr = const I_0
        -->
        I_0 = P_0 / const

        :return: I_0
        """
        return self._p_0 / (pi * self.__r_0**2 * gamma(self._M+1))

    @staticmethod
    @jit(nopython=True)
    def __initialize_field(M, r_0, dr, n_r):
        """
        :param M: power of polynomial before exponent in initial condition
        :param r_0: characteristic spatial size
        :param dr: spatial grid step
        :param n_r: number of points in spatial grid

        :return: initialized field array
        """
        arr = zeros(shape=(n_r,), dtype=complex64)
        for i in range(n_r):
            r = i * dr
            arr[i] = (r / r_0)**M * exp(-0.5 * (r / r_0)**2)

        return arr

    def save_field(self, path):
        field_xy = r_to_xy_complex(self._field)
        save(path, field_xy)
