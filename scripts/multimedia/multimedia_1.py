from numpy import meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, BeamVisualizer


class BeamVisualizer1(BeamVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_beam(self, beam, z, step):
        """Plots intensity distribution in 2D beam with contour_plot"""

        # FLAGS
        ticks = True
        labels = True
        title = True

        fig = plt.figure(figsize=self._fig_size)
        ax = fig.add_subplot(111, projection='3d')

        levels_plot, _ = self._initialize_levels_plot()
        arr, xs, ys = self._initialize_arr()

        xs, ys = [e * 10**6 for e in xs], [e * 10**6 for e in ys]
        xx, yy = meshgrid(xs, ys)
        ax.plot_surface(xx, yy, arr, cmap=self._cmap, rstride=1, cstride=1, antialiased=False,
                        vmin=levels_plot[0], vmax=levels_plot[-1])

        ax.view_init(elev=50, azim=345)

        if beam.info == 'beam_r':
            offset_x = -1.1 * self.__x_max * 10**6
            offset_y = 1.1 * self.__y_max * 10**6
            ax.contour(xx, yy, arr, 1, zdir='x', colors='black', linestyles='solid', linewidths=3, offset=offset_x,
                       levels=1)
            ax.contour(xx, yy, arr, 1, zdir='y', colors='black', linestyles='solid', linewidths=3, offset=offset_y,
                       levels=1)

        if ticks:
            plt.xticks([int(e) for e in self._y_ticklabels], [e + '      ' for e in self._y_ticklabels],
                       fontsize=self._font_size['ticks'])
            plt.yticks([int(e) for e in self._x_ticklabels], [e for e in self._y_ticklabels],
                       fontsize=self._font_size['ticks'])

            n_z_ticks = 3
            di0 = levels_plot[-1] / n_z_ticks
            prec = 2
            zticks = [int(i * di0 * 10 ** prec) / 10 ** prec for i in range(n_z_ticks + 1)]
            ax.set_zlim([levels_plot[0], levels_plot[-1]])
            ax.set_zticks(zticks)
            ax.tick_params(labelsize=self._font_size['ticks'])
            ax.xaxis.set_tick_params(pad=5)
            ax.yaxis.set_tick_params(pad=5)
            ax.zaxis.set_tick_params(pad=30)
        else:
            plt.xticks([])
            plt.yticks([])
            ax.set_zticks([])

        if labels:
            plt.xlabel('\n\n\n\n' + self._y_label, fontsize=self._font_size['labels'],
                       fontweight=self._font_weight['labels'])
            plt.ylabel('\n\n' + self._x_label, fontsize=self._font_size['labels'],
                       fontweight=self._font_weight['labels'])
            if self._normalize_intensity_to == beam.i_0:
                z_label = '$\qquad\qquad\quad$ I/I$\mathbf{_0}$'
            else:
                z_label = '$\qquad\qquad\qquad\mathbf{I}$\n$\qquad\qquad\quad$TW/\n$\quad\qquad\qquad$cm$\mathbf{^2}$'
            ax.text(0, 0, levels_plot[-1] * 0.8, s=z_label, fontsize=self._font_size['labels'],
                    fontweight=self._font_weight['labels'])

        if title:
            plt.title('z = %4.02f cm\nI$_{max}$ = %4.02f TW/cm$^2$\n' %
                      (round(z * 10 ** 2, 3), beam.i_max / 10 ** 16), fontsize=self._font_size['title'])

        bbox = fig.bbox_inches.from_bounds(1.1, 0.3, 10.3, 10.0)

        plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches=bbox)
        plt.close()

        del arr


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

                visualizer = BeamVisualizer1(beam=beam,
                                             maximum_intensity=4*10**16,
                                             normalize_intensity_to=1,
                                             plot_type=None)

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
