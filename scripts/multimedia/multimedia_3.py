from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, BeamVisualizer


class Multimedia3(BaseMultimedia):
    """Gaussian, ring and vortex beams in volume"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, (M, m) in enumerate([(0, 0), (1, 0), (1, 1)]):
            for idx_row, _ in enumerate([0]):
                print('================================================================')
                print('M = %d' % M, ', m = %d' % m)
                print('================================================================')

                p_rel_gauss = 4 if M == 0 and m == 0 else 4.8

                beam = BeamR(medium='LiF',
                             p_0_to_p_gauss=p_rel_gauss,
                             p_0_to_p_vortex=4,
                             m=m,
                             M=M,
                             lmbda=1800 * 10 ** -9,
                             r_0=100 * 10 ** -6,
                             n_r=2048)

                visualizer = BeamVisualizer(beam=beam,
                                            maximum_intensity='local',
                                            normalize_intensity_to=1,
                                            plot_type='volume',
                                            dpi=40)

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=SweepDiffractionExecutorR(beam=beam),
                                        kerr_effect=KerrExecutorR(beam=beam),
                                        n_z=4500,
                                        dz_0=10**-5,
                                        const_dz=True,
                                        print_current_state_every=10,
                                        max_intensity_to_stop=4*10**16,
                                        plot_beam_every=10,
                                        visualizer=visualizer)
                propagator.propagate()

                del beam
                del propagator

                indices.append((idx_col, idx_row))

        all_files, n_pictures_max = self._get_files(self._results_dir)

        return all_files, indices, n_pictures_max


multimedia = Multimedia3()
multimedia.process_multimedia()
