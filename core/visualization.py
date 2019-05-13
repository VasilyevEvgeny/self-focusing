from numpy import transpose
from matplotlib import pyplot as plt
from pylab import contourf

from .functions import r_to_xy_real, crop_x, calc_ticks_x


def plot_beam(mode, beam, z, step, path, plot_beam_normalization):
    fig_size, x_max, y_max, ticks, labels, title, colorbar, bbox = None, None, None, None, None, None, None, None
    if mode in ('xy', 'r'):
        fig_size = (12, 10)
        x_max = 250
        y_max = 250
        title = False
        ticks = True
        labels = True
        colorbar = True
        bbox = 'tight'
    elif mode in ('multimedia'):
        fig_size = (3, 3)
        x_max = 250
        y_max = 250
        title = False
        ticks = False
        labels = False
        colorbar = False

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
    if type(plot_beam_normalization) == float:
        max_intensity_value = plot_beam_normalization
    elif plot_beam_normalization == 'local':
        max_intensity_value = beam.i_max
    di = max_intensity_value / (n_plot_levels)
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

    if mode in ('multimedia'):
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


def plot_noise_field(beam, path):
    x_left = -400 * 10 ** -6
    x_right = 400 * 10 ** -6
    y_left = -400 * 10 ** -6
    y_right = 400 * 10 ** -6

    arr = beam.noise.noise_field_norm
    xs, ys = beam.xs, beam.ys

    arr, x_idx_left, x_idx_right = crop_x(arr, xs, x_left, x_right, mode='x')
    arr, y_idx_left, y_idx_right = crop_x(arr, ys, y_left, y_right, mode='y')

    arr = transpose(arr)

    xs = xs[x_idx_left:x_idx_right]
    ys = ys[y_idx_left:y_idx_right]

    font_size = 50
    fig, ax = plt.subplots(figsize=(12, 10))

    plt.contourf(arr, cmap='gray', levels=100)

    x_labels = ['-200', '0', '+200']
    y_labels = ['-200', '0', '+200']
    x_ticks = calc_ticks_x(x_labels, xs)
    y_ticks = calc_ticks_x(y_labels, ys)
    plt.xticks(x_ticks, y_labels, fontsize=font_size - 5)
    plt.xlabel('x, $\mathbf{\mu}$m', fontsize=font_size, fontweight='bold')
    plt.yticks(y_ticks, x_labels, fontsize=font_size - 5)
    plt.ylabel('y, $\mathbf{\mu}$m', fontsize=font_size, fontweight='bold')

    ax.grid(color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_aspect('equal')

    plt.savefig(path + '/noise_field_norm.png', bbox_inches='tight')
    plt.close()

    del arr


def plot_autocorrelations(beam, path):
    xx_s = [(i * beam.dx - beam.x_max) * 10**6 for i in range(2 * beam.n_x - 1)]
    yy_s = [(i * beam.dy - beam.y_max) * 10**6 for i in range(2 * beam.n_y - 1)]

    r_corr = beam.noise.r_corr_in_meters * 10 ** 6

    autocorr_x, autocorr_y = beam.noise.autocorrs
    example_profile_x, example_profile_y = beam.noise.noise_field_summ[beam.n_x//2,:], \
                                           beam.noise.noise_field_summ[:,beam.n_y//2]

    font_size = 20
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(2, 2, hspace=0.1, wspace=0.1)

    ax_rx = fig.add_subplot(grid[0, 0])
    ax_rx.plot(xx_s, autocorr_x, color='red', linewidth=1, label='$\\bar{K}_x$')
    ax_rx.set_xlim(-0.5 * r_corr, 3.5 * r_corr)

    ax_ry = fig.add_subplot(grid[0, 1])
    ax_ry.plot(yy_s, autocorr_y, color='red', linewidth=1, label='$\\bar{K}_x$')
    ax_ry.set_xlim(-0.5 * r_corr, 3.5 * r_corr)

    ax_x = fig.add_subplot(grid[1, 0])
    ax_x.plot(beam.xs, example_profile_x, color='black', linewidth=1, label='$\\bar{K}_x$')

    ax_y = fig.add_subplot(grid[1, 1])
    ax_y.plot(beam.ys, example_profile_y, color='black', linewidth=1, label='$\\bar{K}_x$')

    #plt.plot(beam.autocorrelation_y, color='blue', linewidth=5, alpha=0.5, label='$\\bar{K}_y$')

    #plt.axvline((idx_left_x * beam.dx - beam.x_max) * 10**6, color='red', linewidth=2, linestyle='dotted')
    #plt.axvline((idx_right_x * beam.dx) * 10**6, color='red', linewidth=2, linestyle='dotted')
    #plt.axvline((idx_left_y * beam.dy - beam.y_max) * 10 ** 6, color='blue', linewidth=2, linestyle='dotted')
    #plt.axvline((idx_right_y * beam.dy) * 10 ** 6, color='blue', linewidth=2, linestyle='dotted')

    #plt.xticks(fontsize=font_size)
    #plt.yticks(fontsize=font_size)

    fig.suptitle('$\mathbf{\sigma_{expected}}$ = %.2f\n$\mathbf{\sigma_{real}}$       = %.2f' %
                 (beam.noise.variance_expected, beam.noise.variance_real), fontsize=font_size, fontweight='bold')

    #plt.xlabel('$\mathbf{\Delta}$, мкм', fontsize=font_size, fontweight='bold')
    #plt.ylabel('$\mathbf{\\bar{K}}$', fontsize=font_size, fontweight='bold')

    #plt.grid(linestyle='dotted', linewidth=2, alpha=0.5)

    #plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), fontsize=font_size, loc='center', ncol=2)

    plt.savefig(path + '/autocorrelations.png', bbox_inches='tight')
    plt.close()


def plot_track(states_arr, parameter_index, path):
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
