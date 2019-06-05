from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, get_files


class Multimedia1(BaseMultimedia):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self):
        indices = []
        for idx_col, noise_percent in enumerate([1]):
            for idx_row, (M, m) in enumerate([(1, 1), (2,2)]):
                print('================================================================')
                print('noise_percent = %02d' % noise_percent, ', M = %d' % M, ', m = %d' % m)
                print('================================================================')

                noise = GaussianNoise(r_corr_in_meters=10 * 10 ** -6,
                                      variance=1)

                beam = BeamXY(medium='SiO2',
                              p_0_to_p_V=5,
                              p_0_to_p_G=5,
                              M=M,
                              m=m,
                              noise_percent=noise_percent,
                              noise=noise,
                              lmbda=1800 * 10 ** -9,
                              x_0=100 * 10 ** -6,
                              y_0=100 * 10 ** -6,
                              n_x=256,
                              n_y=256)

                propagator = Propagator(args=self._args,
                                        multidir_name=self._results_dir_name,
                                        beam=beam,
                                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                                        kerr_effect=KerrExecutorXY(beam=beam),
                                        n_z=2000,
                                        dz0=10 ** -4,
                                        flag_const_dz=True,
                                        dn_print_current_state=50,
                                        dn_plot_beam=50,
                                        beam_in_3D=True)

                propagator.propagate()

                del beam
                del propagator

                indices.append((idx_col, idx_row))

        all_files, n_pictures_max = get_files(self._results_dir)

        return all_files, indices, n_pictures_max

    def process_multimedia(self):
        all_files, indices, n_pictures_max = self._get_data()
        self._compose(all_files, indices, n_pictures_max)


multimedia = Multimedia1()
multimedia.process_multimedia()
