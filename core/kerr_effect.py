from core.libs import *


class KerrExecutor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.beam = kwargs["beam"]
        self.nonlin_phase_const = -1j * self.beam.medium.k_0 * self.beam.medium.n_2 * self.beam.i_0 / self.beam.medium.n_0

    @abc.abstractmethod
    def info(self):
        """Information about beam"""


class KerrExecutor_R(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return "kerr_executor_r"

    @staticmethod
    @jit(nopython=True, debug=True)
    def phase_multiplication(field, intensity, current_nonlin_phase, n_r):
        for i in range(n_r):
            field[i] *= exp(current_nonlin_phase * intensity[i])

        return field

    def process(self, dz):
        current_nonlin_phase = self.nonlin_phase_const * dz
        n_r = self.beam.field.shape[0]
        self.beam.field = self.phase_multiplication(self.beam.field, self.beam.intensity, current_nonlin_phase, n_r)


class KerrExecutor_XY(KerrExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def info(self):
        return "kerr_executor_xy"

    @staticmethod
    @jit(nopython=True)
    def phase_multiplication(field, intensity, current_nonlin_phase, n_x, n_y):
        for i in range(n_x):
            for j in range(n_y):
                field[i, j] *= exp(current_nonlin_phase * intensity[i, j])

        return field

    def process(self, dz):
        current_nonlin_phase = self.nonlin_phase_const * dz
        n_x, n_y = self.beam.field.shape[0], self.beam.field.shape[1]
        self.beam.field = self.phase_multiplication(self.beam.field, self.beam.intensity, current_nonlin_phase, n_x, n_y)
