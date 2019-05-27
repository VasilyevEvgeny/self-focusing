import abc
from numpy import sqrt, exp, arctan
from numpy.linalg import norm
from core.medium import Medium
from core.m_constants import M_Constants


class Analytics(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._m_constants = M_Constants()
        self._lmbda = kwargs['lmbda']

    @abc.abstractmethod
    def calculate_max_intensity(self, z):
        """Calculates max intensity of beam"""


class Diffraction2d(Analytics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__x_0 = kwargs['x_0']
        self._medium = Medium(name=kwargs['medium'],
                              lmbda=self._lmbda,
                              m_constants=self._m_constants)

    def calculate_max_intensity(self, z, x=0):
        z_rel = z / (self._medium.k_0 * self.__x_0**2)
        a_0 = exp(-x**2 / (self.__x_0**2 * (1 + z_rel**2))) / (1.0 + z_rel**2)**0.25
        k_psi = (x / self.__x_0)**2 * z_rel / sqrt((1 + z_rel**2)) - arctan(z_rel)

        return norm(a_0 * exp(1j * k_psi))**2
