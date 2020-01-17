from numpy import angle

from .spectrum import Spectrum


class SpectrumXY(Spectrum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, beam):
        # intensity
        self._intensity_xy = beam.intensity

        # full phase
        self._phase_xy = angle(beam._field)

        # spectrum
        self._make_fft(beam._field)
        self._spectrum_intensity_xy = beam._field_to_intensity(self._spectrum_xy)
