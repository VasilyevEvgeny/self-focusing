import abc
from numba import jit
from numpy import exp, multiply


class KerrExecutor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._beam = kwargs['beam']
        self._nonlin_phase_const = -0.5j * self._beam.r_kerr / self._beam.z_diff

    @abc.abstractmethod
    def info(self):
        """Information about KerrExecutor type"""

    @staticmethod
    @jit(nopython=True)
    def phase_increment(field, intensity, current_nonlin_phase):
        return multiply(field, exp(current_nonlin_phase * intensity))

    def process_kerr_effect(self, dz):
        self._beam._field = self.phase_increment(self._beam._field, self._beam.intensity, self._nonlin_phase_const * dz)


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
