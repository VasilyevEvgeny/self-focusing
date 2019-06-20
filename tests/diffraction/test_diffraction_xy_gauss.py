from core import BeamXY, Propagator, FourierDiffractionExecutorXY, BeamVisualizer, xlsx_to_df
from tests.diffraction.test_diffraction import TestDiffraction

NAME = 'diffraction_xy_gauss'


class TestDiffractionXYGauss(TestDiffraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._add_prefix(NAME)

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

        visualizer = BeamVisualizer(beam=beam,
                                    maximum_intensity='local',
                                    normalize_intensity_to=beam.i_0,
                                    plot_type='volume')

        propagator = Propagator(args=self._args,
                                beam=beam,
                                diffraction=FourierDiffractionExecutorXY(beam=beam),
                                n_z=self._n_z,
                                dz_0=beam.z_diff / self._n_z,
                                const_dz=True,
                                print_current_state_every=0,
                                plot_beam_every=0,
                                visualizer=visualizer)

        propagator.propagate()

        return propagator.logger.track_filename, propagator.manager.results_dir, propagator.beam.z_diff

    def test_diffraction_xy_gauss(self):
        track_filename, path_to_save_plot, z_diff = self.process()
        df = xlsx_to_df(track_filename, normalize_z_to=1)

        self._add_analytics_to_df(df)
        self._check(df)

        if self._flag_plot:
            self._plot(df, path_to_save_plot, z_diff)
