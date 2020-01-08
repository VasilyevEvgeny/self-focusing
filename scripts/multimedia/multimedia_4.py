from scripts.multimedia.multimedia_base import BaseMultimedia
from core import GaussianNoise, BeamXY, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, BeamVisualizer


class Multimedia4(BaseMultimedia):
    """Vortex beams with different correlation radii and noise percent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, r_corr in enumerate([10*10**-6, 50*10**-6, 100*10**-6]):
            for idx_row, noise_percent in enumerate([1, 3, 5]):
                print('=====================================================================')
                print('r_corr = %d' % (round(r_corr * 10**6)), ', noise_percent = %d' % noise_percent)
                print('=====================================================================')

                noise = GaussianNoise(r_corr_in_meters=r_corr,
                                      variance=1)

                beam = BeamXY(medium='LiF',
                              p_0_to_p_vortex=3,
                              m=2,
                              M=2,
                              noise_percent=noise_percent,
                              noise=noise,
                              lmbda=1800 * 10 ** -9,
                              radii_in_grid=10,
                              x_0=100 * 10 ** -6,
                              y_0=100 * 10**-6,
                              n_x=1024,
                              n_y=1024)

                visualizer = BeamVisualizer(beam=beam,
                                            maximum_intensity=4*10**16,
                                            normalize_intensity_to=1,
                                            plot_type='flat',
                                            title_string='noise=%d%%\nr$_{corr}$=%d $\mu$m\n\n' % (noise_percent,
                                                                                            round(r_corr * 10 ** 6)),
                                            dpi=30)

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                                        kerr_effect=KerrExecutorXY(beam=beam),
                                        n_z=3000,
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


multimedia = Multimedia4()
multimedia.process_multimedia()
