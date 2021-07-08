from numpy import transpose, meshgrid, zeros, log10, where
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from pylab import contourf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, cm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')

from .functions import r_to_xy_real, crop_x, calc_ticks_x


class BeamVisualizer:
    """Class for plotting beams in profile, flat and volume styles."""

    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__maximum_intensity = kwargs['maximum_intensity']
        self._normalize_intensity_to = kwargs['normalize_intensity_to']
        if self._normalize_intensity_to not in (self.__beam.i_0, 1):
            raise Exception('Wrong normalize_to arg!')
        self.__plot_type = kwargs['plot_type']
        self.__language = kwargs.get('language', 'english')
        self._path_to_save = None

        # font
        self._font_size = {'title': 40, 'ticks': 40, 'labels': 50, 'colorbar_ticks': 40,
                             'colorbar_label': 50}
        self._font_weight = {'title': 'bold', 'ticks': 'normal', 'labels': 'bold', 'colorbar_ticks': 'bold',
                             'colorbar_label': 'bold'}

        # picture
        self._fig_size = (12, 10)
        self._cmap = plt.get_cmap('jet')

        # axes
        self.__x_max = 250 * 10**-6  # m
        self.__y_max = self.__x_max
        self._x_ticklabels = ['-150', '0', '+150']
        self._y_ticklabels = ['-150', '0', '+150']
        self._x_label, self._y_label = self.__initialize_labels()

        # title
        self.__default_title_string = 'z = %05.2f cm\nI$_{max}$ = %05.2f TW/cm$^2$\n'
        self.__title_string = kwargs.get('title_string', self.__default_title_string)

        # bbox
        self.__bbox_width = 10.3
        self.__bbox_height = 10.0

        # picture
        self.__dpi = kwargs.get('dpi', 50)

    @staticmethod
    def __cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    def get_path_to_save(self, path_to_save):
        self._path_to_save = path_to_save

    def __initialize_labels(self):
        if self.__language == 'english':
            x_label = 'x, $\mathbf{\mu m}$'
            y_label = 'y, $\mathbf{\mu m}$'
        else:
            x_label = 'x, мкм'
            y_label = 'y, мкм'

        return x_label, y_label

    def _initialize_arr(self):
        arr, xs, ys = None, None, None
        if self.__beam.info == 'beam_x':
            n = self.__beam.intensity.shape[0]
            arr = zeros(shape=(n, n))
            arr[:] = self.__beam.intensity[:]
            xs, ys = self.__beam.xs, self.__beam.xs
        elif self.__beam.info == 'beam_r':
            arr = r_to_xy_real(self.__beam.intensity)
            xs = [-e for e in self.__beam.rs][::-1][:-1] + self.__beam.rs
            ys = xs
        elif self.__beam.info == 'beam_xy':
            arr = self.__beam.intensity
            xs, ys = self.__beam.xs, self.__beam.ys

        x_left, x_right = -self.__x_max, self.__x_max
        y_left, y_right = -self.__y_max, self.__y_max

        arr, x_idx_left, x_idx_right = crop_x(arr, xs, x_left, x_right, mode='x')
        arr, y_idx_left, y_idx_right = crop_x(arr, ys, y_left, y_right, mode='y')

        if self.__plot_type != 'flat':
            arr = transpose(arr)

        if self._normalize_intensity_to == 1:
            arr *= self.__beam.i_0 / 10**17

        xs = xs[x_idx_left:x_idx_right]
        ys = ys[y_idx_left:y_idx_right]

        return arr, xs, ys

    def _initialize_levels_plot(self, n_plot_levels=100):
        max_intensity = None

        if isinstance(self.__maximum_intensity, int) or isinstance(self.__maximum_intensity, float):
            max_intensity = self.__maximum_intensity
        elif self.__maximum_intensity == 'local':
            max_intensity = self.__beam.i_max

        if self._normalize_intensity_to == self.__beam.i_0:
            max_intensity /= self.__beam.i_0
        else:
            pass
            # max_intensity /= 10**16

        di = max_intensity / n_plot_levels
        levels_plot = [i * di for i in range(n_plot_levels + 1)]

        return levels_plot, max_intensity

    def plot_beam(self, beam, z, step):
        if self.__plot_type == 'profile':
            return self.__plot_beam_profile(beam, z, step)
        elif self.__plot_type == 'flat':
            # return self.__plot_beam_flat(beam, z, step)
            return self.__plot_beam_flat_dissertation(beam, z, step)
        elif self.__plot_type == 'volume':
            return self.__plot_beam_volume(beam, z, step)
        else:
            raise Exception('Wrong "plot_beam_func"!')

    def __plot_beam_profile(self, beam, z, step):
        """Plots intensity distribution in 1D beam with plot"""

        # FLAGS
        ticks = True
        labels = True
        title = True

        fig, ax = plt.subplots(figsize=(10, 8))

        _, max_intensity = self._initialize_levels_plot()
        arr, xs, ys = self._initialize_arr()
        section = arr[:, arr.shape[1]//2]

        plt.plot(section, color='black', linewidth=5, linestyle='solid')

        y_pad = 0.1 * max_intensity
        plt.ylim([-y_pad, max_intensity + y_pad])

        if ticks:
            x_ticklabels = ['-150', '0', '+150']
            x_ticks = calc_ticks_x(x_ticklabels, xs)
            plt.xticks(x_ticks, x_ticklabels,
                       fontsize=self._font_size['ticks'], fontweight=self._font_weight['ticks'])

            n_y_ticks = 7
            di = max_intensity / (n_y_ticks - 1)
            y_ticks = [i * di for i in range(n_y_ticks)]
            y_ticklabels = ['%05.2f' % e for e in y_ticks]
            plt.yticks(y_ticks, y_ticklabels,
                       fontsize=self._font_size['ticks'], fontweight=self._font_weight['ticks'])

        if labels:
            plt.xlabel(self._x_label, fontsize=self._font_size['labels'], fontweight=self._font_weight['labels'])
            if self._normalize_intensity_to == beam.i_0:
                y_label = 'I/I$\mathbf{_0}$'
                ax.text(-0.25 * len(xs), 1.2 * max_intensity, y_label,
                        fontsize=self._font_size['labels'], fontweight=self._font_weight['labels'])
            else:
                y_label = '$\qquad$I,\nTW/cm$\mathbf{^2}$'
                ax.text(-0.35 * len(xs), 1.2 * max_intensity, y_label,
                        fontsize=self._font_size['labels'], fontweight=self._font_weight['labels'])

        if title:
            if self.__title_string == self.__default_title_string:
                plt.title(self.__title_string %
                          (round(z * 10 ** 2, 3), beam.i_max / 10 ** 16), fontsize=self._font_size['title'])
            else:
                plt.title(self.__title_string, fontsize=self._font_size['title'])

        ax.grid(color='gray', linestyle='dotted', linewidth=2, alpha=0.5)

        bbox = fig.bbox_inches.from_bounds(-0.9, -0.8, self.__bbox_width, self.__bbox_height)

        plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches=bbox, dpi=self.__dpi)
        plt.close()

        del arr

    def __plot_beam_flat(self, beam, z, step):
        """Plots intensity distribution in 2D beam with contour_plot"""

        # FLAGS
        ticks = True
        labels = True
        title = True
        colorbar = True

        fig, ax = plt.subplots(figsize=(9, 7))

        levels_plot, max_intensity = self._initialize_levels_plot(n_plot_levels=500)
        arr, xs, ys = self._initialize_arr()

        plot = contourf(arr, cmap=self._cmap, levels=levels_plot)

        if ticks:
            x_ticks = calc_ticks_x(self._x_ticklabels, xs)
            y_ticks = calc_ticks_x(self._y_ticklabels, ys)
            plt.xticks(x_ticks, self._y_ticklabels, fontsize=self._font_size['ticks'])
            plt.yticks(y_ticks, self._x_ticklabels, fontsize=self._font_size['ticks'])
        else:
            plt.xticks([])
            plt.yticks([])

        if labels:
            plt.xlabel(self._x_label, fontsize=self._font_size['labels'], fontweight=self._font_weight['labels'])
            plt.ylabel(self._y_label, fontsize=self._font_size['labels'], fontweight=self._font_weight['labels'],
                       labelpad=-30)

        if title:
            if self.__title_string == self.__default_title_string:
                plt.title((self.__title_string + '\n') %
                          (round(z * 10 ** 2, 3), beam.i_max / 10 ** 16), fontsize=self._font_size['title'])
            else:
                plt.title(self.__title_string, fontsize=self._font_size['title'])

        ax.grid(color='white', linestyle='dotted', linewidth=3, alpha=0.5)
        ax.set_aspect('equal')

        if colorbar:
            n_ticks_colorbar_levels = 4
            dcb = max_intensity / n_ticks_colorbar_levels
            levels_ticks_colorbar = [i * dcb for i in range(n_ticks_colorbar_levels + 1)]
            colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation='vertical', aspect=10, pad=0.05)
            if self._normalize_intensity_to == beam.i_0:
                colorbar_label = 'I/I$\mathbf{_0}$'
                colorbar.set_label(colorbar_label, labelpad=-60, y=1.25, rotation=0,
                                   fontsize=self._font_size['colorbar_label'],
                                   fontweight=self._font_weight['colorbar_label'])
            else:
                colorbar_label = 'I,\nTW/cm$\mathbf{^2}$'
                colorbar.set_label(colorbar_label, labelpad=-100, y=1.4, rotation=0,
                                   fontsize=self._font_size['colorbar_label'],
                                   fontweight=self._font_weight['colorbar_label'])

            ticks_cbar = ['%05.2f' % e if e != 0 else '00.00' for e in levels_ticks_colorbar]

            colorbar.ax.set_yticklabels(ticks_cbar)
            colorbar.ax.tick_params(labelsize=self._font_size['colorbar_ticks'])

        bbox = fig.bbox_inches.from_bounds(-0.8, -1.0, self.__bbox_width, self.__bbox_height)

        plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches=bbox, dpi=self.__dpi)
        plt.close()

        del arr

    @staticmethod
    def __calc_ticks_x(labels, xs):
        ticks = []
        nxt = 0
        for label in labels:
            for i in range(nxt, len(xs)):
                if xs[i] > float(label.replace('$-$', '-')) * 1e-3:
                    ticks.append(i)
                    nxt = i
                    break
        return ticks

    def __plot_beam_flat_dissertation(self, beam, z, step, legend=True):
        """Plots intensity distribution in 2D beam with contour_plot"""

        w, h = (8.5, 7) if legend else (5, 5)

        fig, ax = plt.subplots(figsize=self.__cm2inch(w, h))
        fig.patch.set_facecolor('white')

        arr, xs, ys = self._initialize_arr()

        # print(np.max(arr) * self.__beam.i_0, self.__maximum_intensity)

        cmap = plt.get_cmap('jet')
        # levels_plot = [-1. + i * 0.005 for i in range(500)]
        levels_plot, max_intensity = self._initialize_levels_plot()
        # levels_plot = [e / self.__beam.i_0 for e in levels_plot]
        # levels_plot = [i * di for i in range(n_levels + 1)]
        # print(self.__beam.i_max)
        # print(levels_plot)
        contour_plot = contourf(arr, cmap=cmap, levels=levels_plot)

        x_ticklabels = ['$-$0.2', '$-$0.1', '0', '+0.1', '+0.2']
        x_ticks = self.__calc_ticks_x(x_ticklabels, xs)
        y_ticklabels = ['$-$0.2', '$-$0.1', '0', '+0.1', '+0.2']
        y_ticks = self.__calc_ticks_x(y_ticklabels, ys)

        ax.tick_params(direction='in', colors='white', labelcolor='black', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        plt.xticks(x_ticks, x_ticklabels, fontsize=10)
        plt.yticks(y_ticks, y_ticklabels, fontsize=10)
        plt.xlabel('$x$, мм', fontsize=12)
        plt.ylabel('$y$, мм', fontsize=12)

        # ax.set_xticklabels(c='black')

        # plt.grid(c='white', ls=':', lw=0.5, alpha=0.5)

        # # t profile
        # t_profile_minimum = np.min(t_profile)
        # t_profile_maximum = np.max(t_profile)
        # delta_t_profile = t_profile_maximum - t_profile_minimum
        # ax_profile = ax.twinx()
        # ax_profile.set_yticks([])
        # ax_profile.plot(t_profile, c='white', lw=1)
        # ax_profile.set_ylim(t_profile_minimum - delta_t_profile * 0.1, t_profile_maximum + delta_t_profile * 5)

        if legend:
        #     levels_cbar = [-1.0, 0.0, 1.0]
            colorbar = fig.colorbar(contour_plot, orientation='vertical', aspect=10, pad=0.05)
            colorbar_label = 'I, TW/cm$\mathbf{^2}$'
            colorbar.set_label(colorbar_label, labelpad=-15, y=1.15, rotation=0,
                               fontsize=12, fontweight='bold')
        #     cbar.set_label('lg$(I/I_0)$', labelpad=-25, y=1.15, rotation=0, fontsize=14)
        #     ticks_cbar = ['+' + str(round(e, 1)) if e > 0 else '$-$' + str(abs(round(e, 1))) if e != 0 else '0.0' for e
        #                   in
        #                   levels_cbar]
        #     cbar.ax.set_yticklabels(ticks_cbar, usetex=True)
        #     cbar.ax.tick_params(labelsize=12)

        bbox = 'tight' if legend else fig.bbox_inches.from_bounds(-0.42, -0.25, 2.25, 2)
        # t_type, number = file.split('/')[0], file.split('/')[1]
        # plt.savefig('{}/{}_{}.png'.format(t_type, t_type, number), bbox_inches=bbox, dpi=500)
        # plt.close()

        plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches=bbox, dpi=500)
        plt.close()

        del arr

    def __plot_beam_volume(self, beam, z, step):
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
            zticks = [i * di0 for i in range(n_z_ticks + 1)]
            zticklabels = ['%05.2f' % (int(e * 10 ** prec) / 10 ** prec) for e in zticks]
            ax.set_zlim([levels_plot[0], levels_plot[-1]])
            ax.set_zticks(zticks)
            ax.set_zticklabels(zticklabels)
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
            if self.__title_string == self.__default_title_string:
                plt.title(self.__title_string %
                          (round(z * 10 ** 2, 3), beam.i_max / 10 ** 16), fontsize=self._font_size['title'])
            else:
                plt.title(self.__title_string, fontsize=self._font_size['title'])

        bbox = fig.bbox_inches.from_bounds(1.1, 0.3, self.__bbox_width, self.__bbox_height)

        plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches=bbox, dpi=self.__dpi)
        plt.close()

        del arr


def plot_noise(beam, path):
    """Plots picture with information about generated complex noise"""

    xx_s = [(i * beam.dx - 0.5 * beam.x_max) * 10**6 for i in range(beam.n_x)]
    yy_s = [(i * beam.dy - 0.5 * beam.y_max) * 10**6 for i in range(beam.n_y)]

    r_corr = beam.noise.r_corr_in_meters * 10 ** 6
    variance_expected = beam.noise.variance_expected
    autocorr_real_x, autocorr_real_y, autocorr_imag_x, autocorr_imag_y = beam.noise.autocorrs

    field_real, field_imag = beam.noise._noise_field_real, beam.noise._noise_field_imag

    font_size = 20
    font_weight = 'bold'
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(4, 2, hspace=0.7, wspace=0.2)

    x_left = -400 * 10 ** -6
    x_right = 400 * 10 ** -6
    y_left = -400 * 10 ** -6
    y_right = 400 * 10 ** -6
    xs, ys = beam.xs, beam.ys

    field_real, x_idx_left, x_idx_right = crop_x(field_real, xs, x_left, x_right, mode='x')
    field_real, y_idx_left, y_idx_right = crop_x(field_real, ys, y_left, y_right, mode='y')
    field_real = transpose(field_real)

    field_imag, x_idx_left, x_idx_right = crop_x(field_imag, xs, x_left, x_right, mode='x')
    field_imag, y_idx_left, y_idx_right = crop_x(field_imag, ys, y_left, y_right, mode='y')
    field_imag = transpose(field_imag)

    xs = xs[x_idx_left:x_idx_right]
    ys = ys[y_idx_left:y_idx_right]

    x_labels = ['-200', '0', '+200']
    y_labels = ['-200', '0', '+200']
    x_ticks = calc_ticks_x(x_labels, xs)
    y_ticks = calc_ticks_x(y_labels, ys)

    # fields

    ax_fr = fig.add_subplot(grid[:2, 0])
    ax_fr.contourf(field_real, cmap='gray', levels=50)
    ax_fr.set_xticks(x_ticks)
    ax_fr.set_xticklabels(y_labels, fontsize=font_size - 5, fontweight=font_weight)
    ax_fr.set_xlabel('x, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_fr.set_yticks(y_ticks)
    ax_fr.set_yticklabels(x_labels, fontsize=font_size - 5, fontweight=font_weight)
    ax_fr.set_ylabel('y, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_fr.grid(color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax_fr.set_aspect('equal')
    ax_fr.set_title('$\mathbf{\sigma^2_{real}}$ = %.2f\n' % beam.noise.variance_real, fontsize=font_size,
                    fontweight=font_weight)

    ax_fi = fig.add_subplot(grid[:2, 1])
    ax_fi.contourf(field_imag, cmap='gray', levels=50)
    ax_fi.set_xticks(x_ticks)
    ax_fi.set_xticklabels(y_labels, fontsize=font_size - 5, fontweight=font_weight)
    ax_fi.set_xlabel('x, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_fi.set_yticks(y_ticks)
    ax_fi.set_yticklabels(x_labels, fontsize=font_size - 5, fontweight=font_weight)
    ax_fi.set_ylabel('y, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_fi.grid(color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax_fi.set_aspect('equal')
    ax_fi.set_title('$\mathbf{\sigma^2_{imag}}$ = %.2f\n' % beam.noise.variance_imag, fontsize=font_size,
                    fontweight=font_weight)

    # autocorrs

    x_min, x_max = 0, 5 * r_corr
    n_xs = 5
    dxs = (x_max - x_min) / n_xs
    x_ticks = [round(i * dxs) for i in range(n_xs + 1)]

    y_min, y_max = -0.15 * variance_expected, variance_expected * 1.15
    n_ys = 5
    dys = variance_expected / (n_ys - 1)
    prec = 2
    y_ticks = [round(int(i * dys * 10**prec) / 10**prec, 2) for i in range(n_ys)]
    y_labels = [ "%.02f" % e for e in y_ticks ]

    ax_rx = fig.add_subplot(grid[2, 0])
    ax_rx.plot(yy_s, autocorr_real_x, color='red', linewidth=5, zorder=2)
    ax_rx.axvline(r_corr, linestyle='solid', color='black', linewidth=3, zorder=1)
    ax_rx.grid(linewidth=1, linestyle='dotted')
    ax_rx.set_xlim(x_min, x_max)
    ax_rx.set_ylim(y_min, y_max)
    ax_rx.set_title('\n$\mathbf{R^x_{real}}$', fontsize=font_size, fontweight=font_weight)
    ax_rx.set_xlabel('y, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_rx.set_xticks(x_ticks)
    ax_rx.set_xticklabels(x_ticks, fontsize=font_size - 5, fontweight=font_weight)
    ax_rx.set_yticks(y_ticks)
    ax_rx.set_yticklabels(y_labels, fontsize=font_size - 5, fontweight=font_weight)

    ax_ix = fig.add_subplot(grid[2, 1])
    ax_ix.plot(yy_s, autocorr_imag_x, color='red', linewidth=5, label='$\\bar{K}_x$', zorder=2)
    ax_ix.axvline(r_corr, linestyle='solid', color='black', linewidth=3, zorder=1)
    ax_ix.grid(linewidth=1, linestyle='dotted')
    ax_ix.set_xlim(x_min, x_max)
    ax_ix.set_ylim(y_min, y_max)
    ax_ix.set_title('\n$\mathbf{R^x_{imag}}$', fontsize=font_size, fontweight=font_weight)
    ax_ix.set_xlabel('y, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_ix.set_xticks(x_ticks)
    ax_ix.set_xticklabels(x_ticks, fontsize=font_size - 5,  fontweight=font_weight)
    ax_ix.set_yticks(y_ticks)
    ax_ix.set_yticklabels(y_labels, fontsize=font_size - 5, fontweight=font_weight)

    ax_ry = fig.add_subplot(grid[3, 0])
    ax_ry.plot(xx_s, autocorr_real_y, color='red', linewidth=5, label='$\\bar{K}_x$', zorder=2)
    ax_ry.axvline(r_corr, linestyle='solid', color='black', linewidth=3, zorder=1)
    ax_ry.grid(linewidth=1, linestyle='dotted')
    ax_ry.set_xlim(x_min, x_max)
    ax_ry.set_ylim(y_min, y_max)
    ax_ry.set_title('\n$\mathbf{R^y_{real}}$', fontsize=font_size, fontweight=font_weight)
    ax_ry.set_xlabel('x, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_ry.set_xticks(x_ticks)
    ax_ry.set_xticklabels(x_ticks, fontsize=font_size - 5, fontweight=font_weight)
    ax_ry.set_yticks(y_ticks)
    ax_ry.set_yticklabels(y_labels, fontsize=font_size - 5, fontweight=font_weight)

    ax_iy = fig.add_subplot(grid[3, 1])
    ax_iy.plot(xx_s, autocorr_imag_y, color='red', linewidth=5, label='$\\bar{K}_x$', zorder=2)
    ax_iy.axvline(r_corr, linestyle='solid', color='black', linewidth=3, zorder=1)
    ax_iy.grid(linewidth=1, linestyle='dotted')
    ax_iy.set_xlim(x_min, x_max)
    ax_iy.set_ylim(y_min, y_max)
    ax_iy.set_title('\n$\mathbf{R^y_{imag}}$', fontsize=font_size, fontweight=font_weight)
    ax_iy.set_xlabel('x, $\mathbf{\mu m}$', fontsize=font_size, fontweight=font_weight)
    ax_iy.set_xticks(x_ticks)
    ax_iy.set_xticklabels(x_ticks, fontsize=font_size - 5, fontweight=font_weight)
    ax_iy.set_yticks(y_ticks)
    ax_iy.set_yticklabels(y_labels, fontsize=font_size - 5, fontweight=font_weight)

    fig.suptitle('Complex Gaussian gaussian_noise $\mathbf{\\xi(x,y) = \\xi_{real}(x,y) + i \\xi_{imag}(x,y)}$\n$\mathbf{\sigma^2_{expected}}$ = %.2f\n$\mathbf{r_{corr} = %d}$ $\mathbf{\mu m}$' %
                 (variance_expected, round(r_corr)), fontsize=font_size, fontweight=font_weight)

    plt.savefig(path + '/gaussian_noise.png', bbox_inches='tight')
    plt.close()

    del field_real, field_imag


def plot_track(states_arr, parameter_index, path):
    """Plots parameter dependence on evolutionary coordinate z"""

    zs = [e * 10 ** 2 for e in states_arr[:, 0]]
    parameters = states_arr[:, parameter_index]

    font_size = 30
    plt.figure(figsize=(15, 5))
    plt.plot(zs, parameters, color='black', linewidth=5, alpha=0.8)

    plt.grid(linestyle='dotted', linewidth=2)

    plt.xlabel('$\mathbf{z}$, cm', fontsize=font_size, fontweight='bold')
    plt.xticks(fontsize=font_size, fontweight='bold')

    plt.ylabel('$\mathbf{I_{max} \ / \ I_0}$', fontsize=font_size, fontweight='bold')
    plt.yticks(fontsize=font_size, fontweight='bold')

    plt.savefig(path + '/i_max(z).png', bbox_inches='tight')
    plt.close()


class SpectrumVisualizer:
    def __init__(self, **kwargs):
        self.__spectrum = kwargs['spectrum']

        self.__log_scale_of_spectrum = kwargs.get('log_scale_of_spectrum', False)
        self._remaining_central_part_coeff_field = kwargs['remaining_central_part_coeff_field']
        self._remaining_central_part_coeff_spectrum = kwargs['remaining_central_part_coeff_spectrum']

    def __crop_arr_field(self, arr):
        """
        :param remaining_central_part_coeff:
        0 -> no points,
        0.5 -> central half number of points,
        1.0 -> all points
        :return:
        """

        if self._remaining_central_part_coeff_field < 0 or self._remaining_central_part_coeff_field > 1:
            raise Exception('Wrong remaining_central part_coeff!')

        n = arr.shape[0]
        delta = int(self._remaining_central_part_coeff_field / 2 * n)
        i_min, i_max = n // 2 - delta, n // 2 + delta

        return arr[i_min:i_max, i_min:i_max]

    def __crop_arr_spectrum(self, arr):
        """
        :param remaining_central_part_coeff:
        0 -> no points,
        0.5 -> central half number of points,
        1.0 -> all points
        :return:
        """

        if self._remaining_central_part_coeff_spectrum < 0 or self._remaining_central_part_coeff_spectrum > 1:
            raise Exception('Wrong remaining_central part_coeff!')

        n = arr.shape[0]
        delta = int(self._remaining_central_part_coeff_spectrum / 2 * n)
        i_min, i_max = n // 2 - delta, n // 2 + delta

        return arr[i_min:i_max, i_min:i_max]

    def get_path_to_save(self, path_to_save):
        self._path_to_save = path_to_save

    @staticmethod
    def __log_spectrum(arr, p=5):
        MAX = np.max(arr)
        low_level = MAX * 10**-p
        arr[where(arr < low_level)] = low_level
        return log10(arr / MAX)

    def plot(self, spectrum, z, step):

        fig = plt.figure(figsize=(15, 10))
        spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

        font_size = 30
        font_weight = 'bold'

        ax1 = fig.add_subplot(spec[0, 0])
        ax1.set_aspect('equal')
        intensity_for_plot = self.__crop_arr_field(spectrum.intensity_xy)
        im1 = ax1.contourf(intensity_for_plot, cmap=plt.get_cmap('jet'), levels=100)
        ax1_ticks = [50, 80, 110]
        ax1.set_xticks(ax1_ticks)
        ax1.set_xticklabels(['$\mathbf{-r_0}$', '0', '$\mathbf{+r_0}$'], fontsize=font_size, fontweight=font_weight)
        ax1.set_yticks(ax1_ticks)
        ax1.set_yticklabels(['$\mathbf{+r_0}$', '0', '$\mathbf{-r_0}$'], fontsize=font_size, fontweight=font_weight)
        ax1.grid(color='white', lw=3, ls=':', alpha=0.5)
        divider = make_axes_locatable(ax1)
        cax1 = divider.new_vertical(size='8%', pad=0.5)
        fig.add_axes(cax1)
        cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
        cb1.set_label('$\mathbf{I(x,y) \ / \ I_{max}}$', labelpad=-90, fontsize=30)
        cb1.set_ticks([np.min(intensity_for_plot), np.max(intensity_for_plot)])
        cb1.set_ticklabels(['0', '1'])
        cb1.ax.tick_params(labelsize=20)


        ax2 = fig.add_subplot(spec[0, 1])
        ax2.set_aspect('equal')
        phase_for_plot = self.__crop_arr_field(spectrum.phase_xy)
        im2 = ax2.contourf(phase_for_plot, cmap=plt.get_cmap('hot'), levels=100)
        ax2_ticks = ax1_ticks
        ax2.set_xticks(ax2_ticks)
        ax2.set_xticklabels(['$\mathbf{-r_0}$', '0', '$\mathbf{+r_0}$'], fontsize=font_size, fontweight=font_weight)
        ax2.set_yticks(ax2_ticks)
        ax2.set_yticklabels(['', '', ''])
        ax2.grid(color='white', lw=3, ls=':', alpha=0.5)
        divider = make_axes_locatable(ax2)
        cax2 = divider.new_vertical(size='8%', pad=0.5)
        fig.add_axes(cax2)
        cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
        cb2.set_label('$\mathbf{\\theta(x,y)}$', labelpad=-90, fontsize=30)
        cb2.set_ticks([np.min(phase_for_plot), np.max(phase_for_plot)])
        cb2.set_ticklabels(['0', '2$\pi$'])
        cb2.ax.tick_params(labelsize=20)

        ax3 = fig.add_subplot(spec[0, 2])
        ax3.set_aspect('equal')
        if self.__log_scale_of_spectrum:
            spectrum_for_plot = self.__log_spectrum(self.__crop_arr_spectrum(spectrum.spectrum_intensity_xy))
        else:
            spectrum_for_plot = self.__crop_arr_spectrum(spectrum.spectrum_intensity_xy)
        im3 = ax3.contourf(spectrum_for_plot, cmap=plt.get_cmap('gray'), levels=100)
        ax3_ticks = ax1_ticks
        ax3.set_xticks(ax3_ticks)
        ax3.set_xticklabels(['$\mathbf{-k_0}$', '0', '$\mathbf{+k_0}$'], fontsize=font_size, fontweight=font_weight)
        ax3.set_yticks(ax3_ticks)
        ax3.yaxis.tick_right()
        ax3.set_yticklabels(['$\mathbf{+k_0}$', '  0', '$\mathbf{-k_0}$'], fontsize=font_size, fontweight=font_weight)
        ax3.grid(color='white', lw=3, ls=':', alpha=0.5)
        divider = make_axes_locatable(ax3)
        cax3 = divider.new_vertical(size='8%', pad=0.5)
        fig.add_axes(cax3)
        cb3 = fig.colorbar(im3, cax=cax3, orientation='horizontal')
        cb3.set_label('$\mathbf{S(k_x,k_y) \ / \ S_{max}}$', labelpad=-90, fontsize=30)
        cb3.set_ticks([np.min(spectrum_for_plot), np.max(spectrum_for_plot)])
        cb3.set_ticklabels(['0', '1'])
        cb3.ax.tick_params(labelsize=20)

        plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches='tight', dpi=50)
        plt.close()

        # fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        # spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
        # ax1 = fig.add_subplot(spec[0, 0])
        # # ax2 = fig.add_subplot(spec[0, 1])
        # ax3 = fig.add_subplot(spec[0, 1])
        # ax4 = fig.add_subplot(spec[0, 2])
        #
        # ax1.set_aspect('equal')
        # # ax2.set_aspect('equal')
        # ax3.set_aspect('equal')
        # ax4.set_aspect('equal')
        #
        # # ax1.set_title('$\mathbf{I(x, y)}$', fontdict={'fontsize': 30})
        # # #ax2.set_title('$\mathbf{\\varphi_{kerr}(x, y)}$', fontdict={'fontsize': 30})
        # # ax3.set_title('$\mathbf{\\varphi(x, y)}$', fontdict={'fontsize': 30})
        # # ax4.set_title('$\mathbf{S(k_x, k_y)}$', fontdict={'fontsize': 30})
        #
        # intensity_for_plot = self.__crop_arr_field(spectrum.intensity_xy)
        # print(intensity_for_plot.shape)
        # # kerr_phase_for_plot = self.__crop_arr_field(spectrum.kerr_phase_xy)
        # phase_for_plot = self.__crop_arr_field(spectrum.phase_xy)
        # if self.__log_scale_of_spectrum:
        #     spectrum_for_plot = self.__log_spectrum(self.__crop_arr_spectrum(spectrum.spectrum_intensity_xy))
        # else:
        #     spectrum_for_plot = self.__crop_arr_spectrum(spectrum.spectrum_intensity_xy)
        #
        # ax1.contourf(intensity_for_plot, cmap=plt.get_cmap('jet'), levels=100)
        # # print('max =', np.max(phase_for_plot))
        # # print('min =', np.min(phase_for_plot))
        # # ax2.contourf(kerr_phase_for_plot, cmap=plt.get_cmap('hot'), levels=100)
        # #ax3.contourf(phase_for_plot, cmap=plt.get_cmap('hot'), levels=100)
        # #ax4.contourf(spectrum_for_plot, cmap=plt.get_cmap('gray'), levels=100)
        #
        # font_size = 30
        # font_weight = 'bold'
        #
        # # #
        # # # propagation
        # # #
        # #
        # # ax1_ticks = [50, 80, 110]
        # # ax1.set_xticks(ax1_ticks)
        # # ax1.set_xticklabels(['$\mathbf{-r_0}$', '0', '$\mathbf{+r_0}$'], fontsize=font_size, fontweight=font_weight)
        # # ax1.set_yticks(ax1_ticks)
        # # ax1.set_yticklabels(['$\mathbf{+r_0}$', '0', '$\mathbf{-r_0}$'], fontsize=font_size, fontweight=font_weight)
        # # ax1.grid(color='white', lw=3, ls=':', alpha=0.5)
        # #
        # # ax3_ticks = ax1_ticks
        # # ax3.set_xticks(ax3_ticks)
        # # ax3.set_xticklabels(['$\mathbf{-r_0}$', '0', '$\mathbf{+r_0}$'], fontsize=font_size, fontweight=font_weight)
        # # ax3.set_yticks(ax1_ticks)
        # # ax3.set_yticklabels(['', ''], fontsize=30)
        # # ax3.grid(color='white', lw=3, ls=':', alpha=0.5)
        # #
        # # ax4_ticks = [50, 80, 110]
        # # ax4.set_xticks(ax4_ticks)
        # # ax4.set_xticklabels(['$\mathbf{-k_0}$', '0', '$\mathbf{+k_0}$'], fontsize=font_size, fontweight=font_weight)
        # # ax4.set_yticks(ax4_ticks)
        # # ax4.yaxis.tick_right()
        # # ax4.set_yticklabels(['$\mathbf{+k_0}$', '  0', '$\mathbf{-k_0}$'], fontsize=font_size, fontweight=font_weight)
        # # ax4.grid(color='white', lw=3, ls=':', alpha=0.5)
        #
        # #
        # # nested vortex
        # #
        #
        # ax1_ticks = [150, 390, 600, 850]
        # ax1.set_xticks(ax1_ticks)
        # ax1.set_xticklabels(['$\mathbf{-\\xi_{out}}$', '$\mathbf{-\\xi_{in}}$', '$\mathbf{+\\xi_{in}}$',
        #                      '$\mathbf{+\\xi_{out}}$'], fontsize=font_size, fontweight=font_weight)
        # ax1.set_yticks(ax1_ticks)
        # ax1.set_yticklabels(['$\mathbf{-\\xi_{out}}$', '$\mathbf{-\\xi_{in}}$', '$\mathbf{+\\xi_{in}}$',
        #                      '$\mathbf{+\\xi_{out}}$'], fontsize=font_size, fontweight=font_weight)
        # ax1.grid(color='white', lw=3, ls=':', alpha=0.5)
        #
        # ax3_ticks = ax1_ticks
        # ax3.set_xticks(ax3_ticks)
        # ax3.set_xticklabels(['$\mathbf{-\\xi_{out}}$', '$\mathbf{-\\xi_{in}}$', '$\mathbf{+\\xi_{in}}$',
        #                      '$\mathbf{+\\xi_{out}}$'], fontsize=font_size, fontweight=font_weight)
        # ax3.set_yticks(ax3_ticks)
        # ax3.set_yticklabels(['', '', '', ''], fontsize=font_size, fontweight=font_weight)
        # ax3.grid(color='white', lw=3, ls=':', alpha=0.5)
        #
        # ax4_ticks = [8, 22, 34, 48]
        # ax4.set_xticks(ax4_ticks)
        # ax4.set_xticklabels(['$\mathbf{-k_2}$', '$\mathbf{-k_1}$', '$\mathbf{+k_1}$', '$\mathbf{+k_2}$'], fontsize=font_size, fontweight=font_weight)
        # ax4.set_yticks(ax4_ticks)
        # ax4.yaxis.tick_right()
        # ax4.set_yticklabels(['$\mathbf{+k_2}$', '$\mathbf{+k_1}$', '$\mathbf{-k_1}$', '$\mathbf{-k_2}$'], fontsize=font_size, fontweight=font_weight)
        # ax4.grid(color='white', lw=3, ls=':', alpha=0.5)
        #
        # # ax1.set_axis_off()
        # # #ax2.set_axis_off()
        # # ax3.set_axis_off()
        # # ax4.set_axis_off()
        #
        # plt.savefig(self._path_to_save + '/%04d.png' % step, bbox_inches='tight', dpi=50)
        # plt.close()

