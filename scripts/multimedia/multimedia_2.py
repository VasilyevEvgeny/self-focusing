from numpy import transpose, meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

from scripts.multimedia.multimedia_base import BaseMultimedia
from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, get_files, \
    r_to_xy_real, crop_x


class Multimedia2(BaseMultimedia):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_data(self, plot_beam_func):
        indices = []
        for idx_row, (M, m) in enumerate([(1, 1)]):
            print('================================================================')
            print('M = %d' % M, ', m = %d' % m)
            print('================================================================')

            beam = BeamR(medium='SiO2',
                         p_0_to_p_V=5,
                         M=M,
                         m=m,
                         lmbda=1800 * 10 ** -9,
                         r_0=100 * 10 ** -6,
                         n_r=2048)

            propagator = Propagator(args=self._args,
                                    multidir_name=self._results_dir_name,
                                    beam=beam,
                                    diffraction=SweepDiffractionExecutorR(beam=beam),
                                    kerr_effect=KerrExecutorR(beam=beam),
                                    n_z=3000,
                                    dz0=10 ** -4,
                                    flag_const_dz=True,
                                    dn_print_current_state=50,
                                    dn_plot_beam=10,
                                    max_intensity_to_stop=5 * beam.i_0,
                                    beam_normalization_type=5,
                                    beam_in_3D=True,
                                    plot_beam_func=plot_beam_func)

            propagator.propagate()

            del beam
            del propagator

            indices.append((0, idx_row))

        all_files, n_pictures_max = get_files(self._results_dir)

        return all_files, indices, n_pictures_max

    @staticmethod
    def plot_beam_func(prefix, beam, z, step, path, plot_beam_normalization):
        fig_size = (12, 10)
        x_max = 250
        y_max = 250
        title = False
        ticks = True
        labels = True

        x_left = -x_max * 10 ** -6
        x_right = x_max * 10 ** -6
        y_left = -y_max * 10 ** -6
        y_right = y_max * 10 ** -6

        arr, xs, ys = None, None, None
        if beam.info == 'beam_r':
            arr = r_to_xy_real(beam.intensity)
            xs = [-e for e in beam.rs][::-1][:-1] + beam.rs
            ys = xs
        elif beam.info == 'beam_xy':
            arr = beam.intensity
            xs, ys = beam.xs, beam.ys

        arr, x_idx_left, x_idx_right = crop_x(arr, xs, x_left, x_right, mode='x')
        arr, y_idx_left, y_idx_right = crop_x(arr, ys, y_left, y_right, mode='y')

        arr = transpose(arr)

        xs = [e * 10 ** 6 for e in xs[x_idx_left:x_idx_right]]
        ys = [e * 10 ** 6 for e in ys[y_idx_left:y_idx_right]]

        xx, yy = meshgrid(xs, ys)

        n_plot_levels = 100
        max_intensity_value = None
        if isinstance(plot_beam_normalization, int) or isinstance(plot_beam_normalization, float):
            max_intensity_value = plot_beam_normalization
        elif plot_beam_normalization == 'local':
            max_intensity_value = beam.i_max
        di = max_intensity_value / n_plot_levels
        levels_plot = [i * di for i in range(n_plot_levels + 1)]

        font_size = 40
        font_weight = 'bold'
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(xx, yy, arr, cmap='jet', rstride=1, cstride=1, antialiased=False,
                        vmin=levels_plot[0], vmax=levels_plot[-1])

        ax.view_init(elev=50, azim=345)

        #offset_x = -1.1 * x_max
        #offset_y = 1.1 * y_max
        #ax.contour(xx, yy, arr, 1, zdir='x', colors='black', linestyles='solid', linewidths=3, offset=offset_x,
        #           levels=1)
        #ax.contour(xx, yy, arr, 1, zdir='y', colors='black', linestyles='solid', linewidths=3, offset=offset_y,
        #           levels=1)

        if ticks:
            x_labels = ['-150', '0', '+150']
            y_labels = ['-150    ', '0    ', '+150    ']
            plt.xticks([int(e) for e in y_labels], y_labels, fontsize=font_size - 5)
            plt.yticks([int(e) for e in x_labels], x_labels, fontsize=font_size - 5)
        else:
            plt.xticks([])
            plt.yticks([])

        ax.set_zlim([levels_plot[0], levels_plot[-1]])
        ax.text(300, 5, 6.5, s='$\qquad\qquad\quad\mathbf{I/I_0}$', fontsize=font_size, fontweight=font_weight)
        n_z_ticks = 4
        di0 = levels_plot[-1] / n_z_ticks
        prec = 2
        zticks = [int(i * di0 * 10 ** prec) / 10 ** prec for i in range(n_z_ticks+1)]
        ax.set_zticks(zticks)

        ax.tick_params(labelsize=font_size - 5)
        ax.xaxis.set_tick_params(pad=10)
        ax.yaxis.set_tick_params(pad=5)
        ax.zaxis.set_tick_params(pad=20)

        if labels:
            plt.xlabel('\n\n\n\nx, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
            plt.ylabel('\n\ny, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)

        if title:
            i_max = beam.i_max * beam.i_0
            plt.title('z = ' + str(round(z * 10 ** 2, 3)) + ' cm\nI$_{max}$ = %.2E' % i_max + ' W/m$^2$\n',
                      fontsize=font_size - 10)

        ax.grid(color='white', linestyle='--', linewidth=3, alpha=0.5)

        bbox = fig.bbox_inches.from_bounds(1.1, 0.3, 10.0, 8.5)

        plt.savefig(path + '/%04d.png' % step, bbox_inches=bbox)
        plt.close()

        del arr

    def process_multimedia(self):
        all_files, indices, n_pictures_max = self._get_data(plot_beam_func=self.plot_beam_func)
        self._compose(all_files, indices, n_pictures_max)


multimedia = Multimedia2()
multimedia.process_multimedia()
