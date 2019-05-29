from core import BeamXY, Propagator, FourierDiffractionExecutorXY, xlsx_to_df
from tests.diffraction.test_diffraction import TestDiffraction

NAME = 'diffraction_xy_gauss'


class TestDiffractionXYGauss(TestDiffraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_prefix(NAME)

        self._p = 1.0
        self._eps = 0.01
        self._png_name = NAME

        self._horizontal_line = 1 / 2

    def process(self):
        beam = BeamXY(medium=self._medium.info,
                      M=0,
                      m=0,
                      p_0_to_p_gauss=self._p_0_to_p_gauss,
                      lmbda=self._lmbda,
                      x_0=self._radius,
                      y_0=self._radius,
                      n_x=256,
                      n_y=256)

        propagator = Propagator(args=self._args,
                                beam=beam,
                                diffraction=FourierDiffractionExecutorXY(beam=beam),
                                n_z=self._n_z,
                                dz0=beam.z_diff / self._n_z,
                                flag_const_dz=True,
                                dn_print_current_state=0,
                                dn_plot_beam=0,
                                beam_normalization_type='local')

        propagator.propagate()

        return propagator.logger.track_filename, propagator.manager.results_dir, propagator.beam.z_diff

    def test_diffraction_xy_gauss(self):
        track_filename, path_to_save_plot, z_diff = self.process()
        df = xlsx_to_df(track_filename, normalize_z_to=1)

        self.add_analytics_to_df(df)
        self.check(df)

        if self._flag_plot:
            self.plot(df, path_to_save_plot, z_diff)
