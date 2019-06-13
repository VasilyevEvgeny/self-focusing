from numpy import transpose, meshgrid
from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from pylab import contourf

from .functions import r_to_xy_real, crop_x, calc_ticks_x


def plot_beam_2d(mode, beam, z, step, path, plot_beam_normalization, x_ticks_normalization_to_x0=True):
    """Plots intensity distribution in 2D beam"""

    fig_size, x_max, ticks, labels, title, bbox = None, None, None, None, None, None
    if mode == 'multimedia':
        fig_size = (3, 3)
        x_max = 400
        title = False
        ticks = False
        labels = False
    else:
        fig_size = (12, 10)
        x_max = 400
        title = False
        ticks = True
        labels = True
        bbox = 'tight'

    font_size = 40
    plt.figure(figsize=fig_size)
    plt.plot(beam.intensity, color='black', linewidth=5)

    if title:
        plt.title('z = ' + str(round(z * 10 ** 2, 3)) + ' cm', fontsize=font_size-10)

    if ticks:

        if x_ticks_normalization_to_x0:
            n_ticks = 11
            x_tickslabels = [ str(i - n_ticks // 2) for i in range(n_ticks) ]
            points_in_x0 = int(beam.x_0 / beam.dx)
            center = int(0.5 * beam.n_x)
            x_ticks = [ center + (i - n_ticks//2) * points_in_x0 for i in range(n_ticks) ]
            plt.xticks(x_ticks, x_tickslabels, fontsize=font_size-10)
        else:
            x_labels = ['-150', '0', '+150']
            x_ticks = calc_ticks_x(x_labels, beam.xs)
            plt.xticks(x_ticks, x_labels, fontsize=font_size-10)

        if isinstance(plot_beam_normalization, int) or isinstance(plot_beam_normalization, float):
            max_intensity_value = plot_beam_normalization
            levels_plot = 7
            di = max_intensity_value / levels_plot
            y_ticks = [i * di for i in range(levels_plot + 1)]
            y_labels = ['%02.02f' % round(e, 2) for e in y_ticks]
            plt.yticks(y_ticks, y_labels, fontsize=font_size - 10)
            percent = 0.15
            plt.ylim([-percent, (1 + percent) * max_intensity_value])
        elif plot_beam_normalization == 'local':
            plt.yticks(fontsize=font_size - 10)

    x_max_ticks = calc_ticks_x([str(-x_max), str(x_max)], beam.xs)
    plt.xlim(x_max_ticks)

    if labels:
        if x_ticks_normalization_to_x0:
            plt.xlabel('$\mathbf{x \ / \ x_0}$', fontsize=font_size, fontweight='bold')
        else:
            plt.xlabel('x, $\mathbf{\mu m}$', fontsize=font_size, fontweight='bold')
        plt.ylabel('$\mathbf{I \ / \ I_0}$', fontsize=font_size, fontweight='bold')

    plt.grid(linestyle='dotted', linewidth=2, alpha=0.5)
    plt.savefig(path + '/%04d.png' % step, bbox_inches='tight')
    plt.close()


def plot_beam_3d_flat(mode, beam, z, step, path, plot_beam_normalization):
    """Plots intensity distribution in 2D beam with contour_plot"""

    fig_size, x_max, y_max, ticks, labels, title, colorbar, bbox = None, None, None, None, None, None, None, None
    if 'multimedia' in mode:
        fig_size = (3, 3)
        x_max = 250
        y_max = 250
        title = False
        ticks = False
        labels = False
        colorbar = False
    else:
        fig_size = (12, 10)
        x_max = 250
        y_max = 250
        title = False
        ticks = True
        labels = True
        colorbar = True
        bbox = 'tight'

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

    xs = xs[x_idx_left:x_idx_right]
    ys = ys[y_idx_left:y_idx_right]

    n_plot_levels = 100
    max_intensity_value = None
    if isinstance(plot_beam_normalization, int) or isinstance(plot_beam_normalization, float):
        max_intensity_value = plot_beam_normalization
    elif plot_beam_normalization == 'local':
        max_intensity_value = beam.i_max
    di = max_intensity_value / n_plot_levels
    levels_plot = [i * di for i in range(n_plot_levels + 1)]

    fig, ax = plt.subplots(figsize=fig_size)
    font_size = 50
    cmap = plt.get_cmap('jet')
    plot = contourf(arr, cmap=cmap, levels=levels_plot)

    if ticks:
        x_labels = ['-150', '0', '+150']
        y_labels = ['-150', '0', '+150']
        x_ticks = calc_ticks_x(x_labels, xs)
        y_ticks = calc_ticks_x(y_labels, ys)
        plt.xticks(x_ticks, y_labels, fontsize=font_size - 5)
        plt.yticks(y_ticks, x_labels, fontsize=font_size - 5)
    else:
        plt.xticks([])
        plt.yticks([])

    if labels:
        plt.xlabel('x, $\mathbf{\mu m}$', fontsize=font_size, fontweight='bold')
        plt.ylabel('y, $\mathbf{\mu m}$', fontsize=font_size, fontweight='bold')

    if title:
        i_max = beam.i_max * beam.i_0
        plt.title('z = ' + str(round(z * 10 ** 2, 3)) + ' cm\nI$_{max}$ = %.2E' % i_max + ' W/m$^2$\n',
                  fontsize=font_size - 10)

    ax.grid(color='white', linestyle='--', linewidth=3, alpha=0.5)
    ax.set_aspect('equal')

    if colorbar:
        n_ticks_colorbar_levels = 4
        dcb = max_intensity_value / n_ticks_colorbar_levels
        levels_ticks_colorbar = [i * dcb for i in range(n_ticks_colorbar_levels + 1)]
        colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation='vertical', aspect=10, pad=0.05)
        colorbar.set_label('I/I$\mathbf{_0}$', labelpad=-140, y=1.2, rotation=0, fontsize=font_size, fontweight='bold')
        ticks_cbar = ['%05.2f' % e if e != 0 else '00.00' for e in levels_ticks_colorbar]
        colorbar.ax.set_yticklabels(ticks_cbar)
        colorbar.ax.tick_params(labelsize=font_size - 10)

    if 'multimedia' in mode:
        bbox = fig.bbox_inches.from_bounds(0.22, 0.19, 2.63, 3)
        if beam.distribution_type == 'gauss':
            m, M = 0, 0
        else:
            m, M = beam.m, beam.M
        C = beam.noise_percent
        plt.title('$M = {:d}, \ m = {:d}, \ C = {:02d}\%$'.format(M, m, C), fontsize=14)

    plt.savefig(path + '/%04d.png' % step, bbox_inches=bbox)
    plt.close()

    del arr


def plot_beam_3d_volume(prefix, beam, z, step, path, plot_beam_normalization):
    """Plots intensity distribution in 2D beam with 3D-plot"""

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

    xs = [e * 10**6 for e in xs[x_idx_left:x_idx_right]]
    ys = [e * 10**6 for e in ys[y_idx_left:y_idx_right]]

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
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xx, yy, arr, cmap='jet', rstride=1, cstride=1, antialiased=False,
                    vmin=levels_plot[0], vmax=levels_plot[-1])

    ax.view_init(elev=50, azim=345)

    offset_x = -1.1 * x_max
    offset_y = 1.1 * y_max
    ax.contour(xx, yy, arr, 1, zdir='x', colors='black', linestyles='solid', linewidths=3, offset=offset_x, levels=0)
    ax.contour(xx, yy, arr, 1, zdir='y', colors='black', linestyles='solid', linewidths=3, offset=offset_y, levels=0)

    if ticks:
        x_labels = ['-150', '0', '+150']
        y_labels = ['-150', '0', '+150']
        plt.xticks([int(e) for e in y_labels], fontsize=font_size - 5)
        plt.yticks([int(e) for e in x_labels], fontsize=font_size - 5)
    else:
        plt.xticks([])
        plt.yticks([])

    ax.set_zlim([levels_plot[0], levels_plot[-1]])
    n_z_ticks = 3
    di0 = levels_plot[-1] / n_z_ticks
    prec = 2
    zticks = [int(i * di0 * 10**prec) / 10**prec for i in range(n_z_ticks)]
    ax.set_zticks(zticks)

    ax.tick_params(labelsize=font_size - 5)
    ax.xaxis.set_tick_params(pad=30)
    ax.yaxis.set_tick_params(pad=5)
    ax.zaxis.set_tick_params(pad=20)

    if labels:
        plt.xlabel('\n\n\n\nx, $\mathbf{\mu m}$', fontsize=font_size, fontweight='bold')
        plt.ylabel('\n\ny, $\mathbf{\mu m}$', fontsize=font_size, fontweight='bold')

    if title:
        i_max = beam.i_max * beam.i_0
        plt.title('z = ' + str(round(z * 10 ** 2, 3)) + ' cm\nI$_{max}$ = %.2E' % i_max + ' W/m$^2$\n',
                  fontsize=font_size - 10)

    ax.grid(color='white', linestyle='--', linewidth=3, alpha=0.5)

    bbox = fig.bbox_inches.from_bounds(1.1, 0.3, 10.0, 8.5)

    plt.savefig(path + '/%04d.png' % step, bbox_inches=bbox)
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
    x_ticks = [int(i * dxs) for i in range(n_xs + 1)]

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

    fig.suptitle('Complex Gaussian gaussian_noise $\mathbf{\\xi(x,y) = \\xi_{real}(x,y) + i \\xi_{imag}(x,y)}$\n$\mathbf{\sigma^2_{expected}}$ = %.2f\n$\mathbf{r^* = %d}$ $\mathbf{\mu m}$' %
                 (variance_expected, round(r_corr)), fontsize=font_size, fontweight=font_weight)

    plt.savefig(path + '/gaussian_noise.png', bbox_inches='tight', dpi=50)
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
