from abc import ABCMeta, abstractmethod
from numba import jit
from numpy import exp, multiply


class KerrExecutor(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__nonlin_phase_const = -0.5j * self.__beam.r_kerr / self.__beam.z_diff

    @abstractmethod
    def info(self):
        """Information about KerrExecutor type"""

    @staticmethod
    @jit(nopython=True)
    def phase_increment(field, intensity, current_nonlin_phase):
        return multiply(field, exp(current_nonlin_phase * intensity))

    def process_kerr_effect(self, dz):
        self.__beam._field = self.phase_increment(self.__beam._field, self.__beam.intensity, self.__nonlin_phase_const * dz)


class KerrExecutorX(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_x'


class KerrExecutorR(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_r'


class KerrExecutorXY(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_xy'
