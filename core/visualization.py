from core.functions import *


def plot_beam(beam, z, step, path, plot_beam_normalization, title=False):
    x_left = -400 * 10 ** -6
    x_right = 400 * 10 ** -6
    y_left = -400 * 10 ** -6
    y_right = 400 * 10 ** -6

    arr, xs, ys = None, None, None
    if beam.info == "beam_r":
        arr = r_to_xy_real(beam.intensity)
        xs = [-e for e in beam.rs][::-1][:-1] + beam.rs
        ys = xs
    elif beam.info == "beam_xy":
        arr = beam.intensity
        xs, ys = beam.xs, beam.ys

    arr, x_idx_left, x_idx_right = crop_x(arr, xs, x_left, x_right, mode="x")
    arr, y_idx_left, y_idx_right = crop_x(arr, ys, y_left, y_right, mode="y")

    arr = np.transpose(arr)

    xs = xs[x_idx_left:x_idx_right]
    ys = ys[y_idx_left:y_idx_right]

    n_plot_levels = 100
    max_intensity_value = None
    if type(plot_beam_normalization) == float:
        max_intensity_value = plot_beam_normalization
    elif plot_beam_normalization == "local":
        max_intensity_value = beam.i_max
    di = max_intensity_value / (n_plot_levels)
    levels_plot = [i * di for i in range(n_plot_levels + 1)]

    fig, ax = plt.subplots(figsize=(12, 10))
    font_size = 50
    cmap = plt.get_cmap("jet")
    plot = contourf(arr, cmap=cmap, levels=levels_plot)
    x_labels = ["-200", "0", "+200"]
    y_labels = ["-200", "0", "+200"]
    x_ticks = calc_ticks_x(x_labels, xs)
    y_ticks = calc_ticks_x(y_labels, ys)
    plt.xticks(x_ticks, y_labels, fontsize=font_size - 5)
    plt.xlabel("x, мкм", fontsize=font_size, fontweight="bold")
    plt.yticks(y_ticks, x_labels, fontsize=font_size - 5)
    plt.ylabel("y, мкм", fontsize=font_size, fontweight="bold")

    if title:
        i_max = np.max(beam.a_to_i()) * beam.i_0
        plt.title("z = " + str(round(z * 10 ** 2, 3)) + " см\nI$_{max}$ = %.2E" % i_max + " Вт/м$^2$\n",
                  fontsize=font_size - 10)

    plt.xlabel("x, мкм")
    plt.ylabel("y, мкм")
    ax.grid(color="white", linestyle='--', linewidth=3, alpha=0.5)
    ax.set_aspect("equal")
    n_ticks_colorbar_levels = 4
    dcb = max_intensity_value / n_ticks_colorbar_levels
    levels_ticks_colorbar = [i * dcb for i in range(n_ticks_colorbar_levels + 1)]
    colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation="vertical", aspect=10, pad=0.05)
    colorbar.set_label("I/I$\mathbf{_0}$", labelpad=-140, y=1.2, rotation=0, fontsize=font_size, fontweight="bold")
    ticks_cbar = ["%05.2f" % e if e != 0 else "00.00" for e in levels_ticks_colorbar]
    colorbar.ax.set_yticklabels(ticks_cbar)
    colorbar.ax.tick_params(labelsize=font_size - 10)

    plt.savefig(path + "/%04d.png" % step, bbox_inches="tight")
    plt.close()

    del arr


def plot_noise_field(beam, path):
    x_left = -400 * 10 ** -6
    x_right = 400 * 10 ** -6
    y_left = -400 * 10 ** -6
    y_right = 400 * 10 ** -6

    arr = beam.noise_field
    xs, ys = beam.xs, beam.ys

    arr, x_idx_left, x_idx_right = crop_x(arr, xs, x_left, x_right, mode="x")
    arr, y_idx_left, y_idx_right = crop_x(arr, ys, y_left, y_right, mode="y")

    arr = np.transpose(arr)

    xs = xs[x_idx_left:x_idx_right]
    ys = ys[y_idx_left:y_idx_right]

    font_size = 50
    fig, ax = plt.subplots(figsize=(12, 10))

    plt.contourf(arr, cmap="gray", levels=100)

    x_labels = ["-200", "0", "+200"]
    y_labels = ["-200", "0", "+200"]
    x_ticks = calc_ticks_x(x_labels, xs)
    y_ticks = calc_ticks_x(y_labels, ys)
    plt.xticks(x_ticks, y_labels, fontsize=font_size - 5)
    plt.xlabel("x, мкм", fontsize=font_size, fontweight="bold")
    plt.yticks(y_ticks, x_labels, fontsize=font_size - 5)
    plt.ylabel("y, мкм", fontsize=font_size, fontweight="bold")

    ax.grid(color="red", linestyle='--', linewidth=2, alpha=0.5)
    ax.set_aspect("equal")

    plt.savefig(path + "/noise_field.png", bbox_inches="tight")
    plt.close()

    del arr


def plot_autocorrelations(beam, path):
    xx_s = [(i * beam.dx - beam.x_max) * 10**6 for i in range(2 * beam.n_x - 1)]
    yy_s = [(i * beam.dy - beam.y_max) * 10**6 for i in range(2 * beam.n_y - 1)]

    r_corr = beam.r_corr_in_meters * 10 ** 6

    font_size = 20
    plt.figure(figsize=(10, 5))
    plt.plot(xx_s, beam.autocorrelation_x, color="red", linewidth=5, alpha=0.5, label="$\\bar{K}_x$")
    plt.plot(yy_s, beam.autocorrelation_y, color="blue", linewidth=5, alpha=0.5, label="$\\bar{K}_y$")

    plt.axvline(-r_corr, color="black", linewidth=2)
    plt.axvline(r_corr, color="black", linewidth=2)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlim([-3.5 * r_corr, 3.5 * r_corr])

    plt.xlabel("x, мкм", fontsize=font_size, fontweight="bold")
    plt.ylabel("$\\bar{K}$", fontsize=font_size, fontweight="bold")

    plt.grid(linestyle="dotted", linewidth=2, alpha=0.5)

    plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), fontsize=font_size, loc="center", ncol=2)

    plt.savefig(path + "/autocorrelations.png", bbox_inches="tight")
    plt.close()


def plot_track(states_arr, parameter_index, path):
    zs = [e * 10 ** 2 for e in states_arr[:, 0]]
    parameters = states_arr[:, parameter_index]

    font_size = 30
    plt.figure(figsize=(15, 5))
    plt.plot(zs, parameters, color="black", linewidth=5, alpha=0.8, label="Щелевой пучок")

    plt.grid(linestyle="dotted", linewidth=2)

    plt.xlabel("$\mathbf{z}$, см", fontsize=font_size, fontweight="bold")
    plt.xticks(fontsize=font_size, fontweight="bold")

    plt.ylabel("$\mathbf{I_{max} \ / \ I_0}$", fontsize=font_size, fontweight="bold")
    plt.yticks(fontsize=font_size, fontweight="bold")

    plt.savefig(path + "/i_max(z).png", bbox_inches='tight')
    plt.close()
