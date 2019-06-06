from abc import ABCMeta, abstractmethod
from numba import jit
from numpy import exp, multiply


class KerrExecutor(metaclass=ABCMeta):
    """
    Abstract class for Kerr effect object.
    The class takes on the input in the constructor a beam object, which contains all the necessary beam parameters
    for further calculations.
    """

    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__nonlin_phase_const = -0.5j * self.__beam.r_kerr / self.__beam.z_diff  # nonlinear Kerr phase shift const

    @abstractmethod
    def info(self):
        """KerrExecutor type"""

    @staticmethod
    @jit(nopython=True)
    def phase_increment(field, intensity, current_nonlin_phase):
        """
        :param field: array for complex light field
        :param intensity: array for float intensity of the field
        :param current_nonlin_phase: current nonlinear phase shift

        :return: field with nonlinear incremented phase shift
        """
        return multiply(field, exp(current_nonlin_phase * intensity))

    def process_kerr_effect(self, dz):
        """
        :param dz: current step along evolutionary coordinate z

        :return: None
        """
        self.__beam._field = self.phase_increment(self.__beam._field, self.__beam.intensity,
                                                  self.__nonlin_phase_const * dz)


class KerrExecutorX(KerrExecutor):
    """
    Class for modeling the Kerr effect to which a 2-dimensional beam is exposed
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_x'


class KerrExecutorR(KerrExecutor):
    """
    Class for modeling the Kerr effect to which a 3-dimensional beam in axisymmetric approximation is exposed
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_r'


class KerrExecutorXY(KerrExecutor):
    """
    Class for modeling the Kerr effect to which a 3-dimensional beam is exposed
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_xy'
