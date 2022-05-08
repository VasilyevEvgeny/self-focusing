from abc import ABCMeta, abstractmethod
from numpy.fft import fft2, fftshift


class Spectrum(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self._beam = kwargs['beam']

        # field
        self._kerr_phase_xy = None
        self._phase_xy = None
        self._nonvortex_phase_xy = None
        self._intensity_xy = None

        # spectrum
        self._spectrum_xy = None
        self._spectrum_intensity_xy = None

    def _make_fft(self, arr):
        self._spectrum_xy = fft2(arr)
        self._spectrum_xy = fftshift(self._spectrum_xy, axes=(0, 1))

    @abstractmethod
    def update(self, beam):
        """"""

    @property
    def beam(self):
        return self._beam

    @property
    def intensity_xy(self):
        return self._intensity_xy

    @property
    def kerr_phase_xy(self):
        return self._kerr_phase_xy

    @property
    def phase_xy(self):
        return self._phase_xy

    @property
    def nonvortex_phase_xy(self):
        return self._nonvortex_phase_xy

    @property
    def spectrum_intensity_xy(self):
        return self._spectrum_intensity_xy
