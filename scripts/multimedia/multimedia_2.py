from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamX, SweepDiffractionExecutorX, KerrExecutorX, Propagator, BeamVisualizer


class Multimedia2(BaseMultimedia):
    """BeamX in 3 representations"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, plot_type in enumerate(['profile', 'flat', 'volume']):
            for idx_row, M in enumerate([1]):
                print('================================================================')
                print('plot_type = %s' % plot_type, ', M = %d' % M)
                print('================================================================')

                beam = BeamX(medium='LiF',
                             M=1,
                             half=False,
                             lmbda=1800 * 10 ** -9,
                             x_0=100 * 10 ** -6,
                             n_x=2048,
                             r_kerr=75.4)

                visualizer = BeamVisualizer(beam=beam,
                                            maximum_intensity=4*10**16,
                                            normalize_intensity_to=beam.i_0,
                                            plot_type=plot_type)

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=SweepDiffractionExecutorX(beam=beam),
                                        kerr_effect=KerrExecutorX(beam=beam),
                                        n_z=2000,
                                        dz_0=10**-5,
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


multimedia = Multimedia2()
multimedia.process_multimedia()
