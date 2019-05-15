import abc
from numba import jit
from numpy import exp, multiply


class KerrExecutor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._beam = kwargs['beam']
        self._nonlin_phase_const = -1j * self._beam.medium.k_0 * self._beam.medium.n_2 * self._beam.i_0 / self._beam.medium.n_0

    @abc.abstractmethod
    def info(self):
        """Information about KerrExecutor type"""

    @staticmethod
    @jit(nopython=True)
    def phase_increment(field, intensity, current_nonlin_phase):
        return multiply(field, exp(current_nonlin_phase * intensity))

    def process_kerr_effect(self, dz):
        self._beam._field = self.phase_increment(self._beam._field, self._beam.intensity, self._nonlin_phase_const * dz)


class KerrExecutor_R(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_r'


class KerrExecutor_XY(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return 'kerr_executor_xy'
