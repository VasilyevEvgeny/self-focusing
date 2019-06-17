from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, BeamVisualizer


class Multimedia1(BaseMultimedia):
    """1 XY vortex with noise"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, noise_percent in enumerate([1]):
            for idx_row, (M, m) in enumerate([(1, 1)]):
                print('================================================================')
                print('noise_percent = %02d' % noise_percent, ', M = %d' % M, ', m = %d' % m)
                print('================================================================')

                noise = GaussianNoise(r_corr=10 * 10 ** -6,
                                      variance=1)

                beam = BeamXY(medium='SiO2',
                              p_0_to_p_vortex=5,
                              M=M,
                              m=m,
                              noise_percent=noise_percent,
                              noise=noise,
                              lmbda=1800 * 10 ** -9,
                              x_0=100 * 10 ** -6,
                              y_0=100 * 10 ** -6,
                              n_x=512,
                              n_y=512,
                              radii_in_grid=20)

                visualizer = BeamVisualizer(beam=beam,
                                            maximum_intensity=4*10**16,
                                            normalize_intensity_to=1,
                                            plot_type='volume')

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                                        kerr_effect=KerrExecutorXY(beam=beam),
                                        n_z=3000,
                                        dz_0=10**-5,
                                        const_dz=True,
                                        print_current_state_every=50,
                                        max_intensity_to_stop=4*10**16,
                                        plot_beam_every=50,
                                        visualizer=visualizer)
                propagator.propagate()

                del beam
                del propagator

                indices.append((idx_col, idx_row))

        all_files, n_pictures_max = self._get_files(self._results_dir)

        return all_files, indices, n_pictures_max


multimedia = Multimedia1()
multimedia.process_multimedia()
