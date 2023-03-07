from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, BeamVisualizer


class Multimedia5(BaseMultimedia):
    """Diffraction of 3 beams"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, (M, m) in enumerate([(0, 0), (1, 1)]):
            for idx_row, plot_type in enumerate(['profile']):
                print('================================================================')
                print('M = %d' % M, ', m = %d' % m)
                print('================================================================')

                p_rel_gauss = 4 if M == 0 and m == 0 else 4.8

                beam = BeamR(medium='LiF',
                             m=m,
                             M=M,
                             p_0_to_p_vortex=1,
                             p_0_to_p_gauss=p_rel_gauss,
                             lmbda=1800 * 10 ** -9,
                             r_0=80 * 10 ** -6,
                             n_r=2048)

                if plot_type == 'profile':
                    visualizer = BeamVisualizer(beam=beam,
                                                maximum_intensity=0.7*10**16,
                                                normalize_intensity_to=beam.i_0,
                                                plot_type=plot_type,
                                                dpi=40)
                else:
                    visualizer = BeamVisualizer(beam=beam,
                                                maximum_intensity=0.7 * 10 ** 16,
                                                normalize_intensity_to=beam.i_0,
                                                plot_type=plot_type,
                                                title_string='',
                                                dpi=40)

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=SweepDiffractionExecutorR(beam=beam),
                                        n_z=1000,
                                        dz_0=beam.z_diff / 1000,
                                        const_dz=True,
                                        print_current_state_every=10,
                                        max_intensity_to_stop=4*10**16,
                                        plot_beam_every=10,
                                        visualizer=visualizer)
                propagator.propagate()

                del beam
                del visualizer
                del propagator

                indices.append((idx_col, idx_row))

        all_files, n_pictures_max = self._get_files(self._results_dir)

        return all_files, indices, n_pictures_max


multimedia = Multimedia5()
multimedia.process_multimedia()
