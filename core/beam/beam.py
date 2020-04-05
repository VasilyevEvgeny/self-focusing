from abc import ABCMeta, abstractmethod
from numba import jit
from numpy import max as maximum

from core.medium import Medium
from core.m_constants import MathConstants


class Beam(metaclass=ABCMeta):
    """
    Abstract class for beam object.
    The class contains all the necessary information about the laser field propagating in the medium.

    All physical quantities are given in the SI system, except for the field A and intensity I.  They are dimensionless
    respectively on A_0 and I_0, where I_0 = c n_0 epsilon_0 |A_0|^2 / 2
    """

    def __init__(self, **kwargs):
        self.__m_constants = MathConstants()  # mathematical constants
        self._lmbda = kwargs['lmbda']  # beam wavelength, [m]

        self._medium = Medium(name=kwargs['medium'],           #
                              lmbda=self._lmbda,               # medium, where beam propagates
                              m_constants=self.__m_constants)  #

        self._M = kwargs['M']  # power of polynomial before exponent in initial condition

        self._distribution_type = None  # type of distribution, depending on M and m:
                                        # M = 0, m = 0  ->  gauss
                                        # M = 0, m > 0  ->  prohibited!
                                        # M > 0, m = 0  ->  ring
                                        # M > 0, m > 0  ->  vortex

        self._radii_in_grid = kwargs.get('radii_in_grid', 20)  # grid_size / radius, [a.u.]

        self._field = None              # array for complex light field
        self._intensity = None          # array for float intensity of the field
        self._i_max = None              # peak beam intensity for z = const, [W/m^2]
        self._i_0 = None                # coefficient for initial beam intensity, [W/m^2]
        self._z_diff = None             # diffraction length of the beam, [m]
                                        # z_diff = k_0 r_0^2
        self._r_kerr = None             # nonlinearity parameter for Kerr effect, [rad]
                                        # r_kerr = 2 k_0 n_2 I_0 z_diff / n_0

    @abstractmethod
    def info(self):
        """Beam type"""

    @abstractmethod
    def save_field(self, path, only_center=True):
        """"""

    def update_intensity(self):
        self._intensity = self._field_to_intensity(self._field)
        self._i_max = maximum(self._intensity) * self._i_0

    @staticmethod
    @jit(nopython=True)
    def _field_to_intensity(field):
        """Intensity calculation as a squared field norm"""
        intensity = field.real**2 + field.imag**2

        return intensity

    @property
    def medium(self):
        return self._medium

    @property
    def lmbda(self):
        return self._lmbda

    @property
    def distribution_type(self):
        return self._distribution_type

    @property
    def M(self):
        return self._M

    @property
    def i_0(self):
        return self._i_0

    @property
    def i_max(self):
        return self._i_max

    @property
    def field(self):
        return self._field

    @property
    def intensity(self):
        return self._intensity

    @property
    def z_diff(self):
        return self._z_diff

    @property
    def r_kerr(self):
        return self._r_kerr
