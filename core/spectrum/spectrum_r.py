from numpy import zeros, complex64, exp, arctan2, pi, angle, save
from numba import jit

from core.functions import r_to_xy_real, r_to_xy_complex
from .spectrum import Spectrum


class SpectrumR(Spectrum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__vortex_phase = self.__initialize_vortex_phase(self._beam.m,
                                                             2 * self._beam.r_max,
                                                             2 * self._beam.n_r,
                                                             self._beam.dr)

    @staticmethod
    @jit(nopython=True)
    def __initialize_vortex_phase(m, perp_max, n_perp, d_perp):
        vortex_phase = zeros((n_perp, n_perp), dtype=complex64)
        for i in range(n_perp):
            for j in range(n_perp):
                x, y = d_perp * i - 0.5 * perp_max, d_perp * j - 0.5 * perp_max
                vortex_phase[i, j] = exp(1j * m * (arctan2(x, y) + pi))

        return vortex_phase

    def update(self, beam):
        # intensity
        self._intensity_xy = r_to_xy_real(beam.intensity)

        # field
        field_xy = r_to_xy_complex(beam._field)

        # kerr phase
        self._kerr_phase_xy = angle(field_xy)

        # full phase
        self._nonvortex_phase_xy = angle(field_xy)
        field_xy *= self.__vortex_phase
        self._phase_xy = angle(field_xy)

        # spectrum
        self._make_fft(field_xy)
        self._spectrum_intensity_xy = beam._field_to_intensity(self._spectrum_xy)

    def save_spectrum(self, path, only_center=True):
        if only_center:
            percent = 15
            center = self.beam.n_r
            # print('self.beam.n_r = ', self.beam.n_r)
            ambit = int(self.beam.n_r * percent / 100)
            # print('ambit = ', ambit)
            spectrum_xy = self._spectrum_xy[center-ambit:center+ambit, center-ambit:center+ambit]
            # print(spectrum_xy.shape)
        else:
            spectrum_xy = self._spectrum_xy
        save(path, spectrum_xy)
