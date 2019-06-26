from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamX, SweepDiffractionExecutorX, KerrExecutorX, Propagator, BeamVisualizer


class Multimedia6(BaseMultimedia):
    """BeamX half in 3D representation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, plot_type in enumerate(['volume']):
            for idx_row, M in enumerate([1]):
                print('================================================================')
                print('plot_type = %s' % plot_type, ', M = %d' % M)
                print('================================================================')

                beam = BeamX(medium='LiF',
                             M=M,
                             half=True,
                             lmbda=1800 * 10 ** -9,
                             x_0=92 * 10 ** -6,
                             n_x=2048,
                             r_kerr=75.4)

                visualizer = BeamVisualizer(beam=beam,
                                            maximum_intensity=5*10**16,
                                            normalize_intensity_to=beam.i_0,
                                            plot_type=plot_type)

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=SweepDiffractionExecutorX(beam=beam),
                                        kerr_effect=KerrExecutorX(beam=beam),
                                        n_z=2000,
                                        dz_0=0.3 * beam.z_diff / 2000,
                                        const_dz=True,
                                        print_current_state_every=10,
                                        max_intensity_to_stop=5*10**16,
                                        plot_beam_every=10,
                                        visualizer=visualizer)
                propagator.propagate()

                del beam
                del propagator

                indices.append((idx_col, idx_row))

        all_files, n_pictures_max = self._get_files(self._results_dir)

        return all_files, indices, n_pictures_max


multimedia = Multimedia6()
multimedia.process_multimedia()
