from core.functions import *


def plot_beam(beam, z, step, path, plot_beam_normalization, title=False):
    x_left = -400 * 10**-6
    x_right = 400 * 10**-6
    y_left = -400 * 10**-6
    y_right = 400 * 10**-6

    arr, xs, ys = None, None, None
    if beam.info == "beam_r":
        arr = r_to_xy_real(beam.intensity)
        xs = [ -e for e in beam.rs ][::-1][:-1] + beam.rs
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
                 fontsize=font_size-10)

    plt.xlabel("x, мкм")
    plt.ylabel("y, мкм")
    ax.grid(color="white", linestyle='--', linewidth=3, alpha=0.5)
    ax.set_aspect("equal")
    n_ticks_colorbar_levels = 4
    dcb = max_intensity_value / n_ticks_colorbar_levels
    levels_ticks_colorbar = [ i * dcb for i in range(n_ticks_colorbar_levels + 1) ]
    colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation="vertical", aspect=10, pad=0.05)
    colorbar.set_label("I/I$\mathbf{_0}$", labelpad=-140, y=1.2, rotation=0, fontsize=font_size, fontweight="bold")
    ticks_cbar = ["%05.2f" % e if e != 0 else "00.00" for e in levels_ticks_colorbar]
    colorbar.ax.set_yticklabels(ticks_cbar)
    colorbar.ax.tick_params(labelsize=font_size-10)

    plt.savefig(path + "/%04d.png" % step, bbox_inches="tight")
    plt.close()

    del arr


def plot_phase_screen(beam, path):
    x_left = -400 * 10 ** -6
    x_right = 400 * 10 ** -6
    y_left = -400 * 10 ** -6
    y_right = 400 * 10 ** -6

    arr = beam.phase_noise_screen.imag
    xs, ys = beam.xs, beam.ys

    arr, x_idx_left, x_idx_right = crop_x(arr, xs, x_left, x_right, mode="x")
    arr, y_idx_left, y_idx_right = crop_x(arr, ys, y_left, y_right, mode="y")

    arr = np.transpose(arr)

    xs = xs[x_idx_left:x_idx_right]
    ys = ys[y_idx_left:y_idx_right]

    #print("median = ", np.median(beam.phase_noise_screen))
    #print("min = ", np.min(beam.phase_noise_screen))
    #print("max = ", np.max(beam.phase_noise_screen))

    font_size = 50
    fig, ax = plt.subplots(figsize=(12, 10))

    const_sigma = 0.3
    med_val = beam.phase_noise_percent / 100 * 2 * pi
    max_val = med_val + const_sigma * med_val
    min_val = med_val - const_sigma * med_val
    n_levels_plot = 100
    dp = (max_val - min_val) / n_levels_plot
    levels_plot = [ min_val + i * dp for i in range(n_levels_plot) ]

    plot = plt.contourf(arr, cmap="jet", levels=levels_plot)

    x_labels = ["-200", "0", "+200"]
    y_labels = ["-200", "0", "+200"]
    x_ticks = calc_ticks_x(x_labels, xs)
    y_ticks = calc_ticks_x(y_labels, ys)
    plt.xticks(x_ticks, y_labels, fontsize=font_size - 5)
    plt.xlabel("x, мкм", fontsize=font_size, fontweight="bold")
    plt.yticks(y_ticks, x_labels, fontsize=font_size - 5)
    plt.ylabel("y, мкм", fontsize=font_size, fontweight="bold")

    ax.grid(color="white", linestyle='--', linewidth=5)
    ax.set_aspect("equal")

    levels_ticks_colorbar = [ med_val - 0.5 * const_sigma * med_val,
                              med_val,
                              med_val + 0.5 * const_sigma * med_val ]

    ticks_cbar = [ "%05.2f" % (beam.phase_noise_percent - 0.5 * const_sigma * beam.phase_noise_percent),
                   "%05.2f" % beam.phase_noise_percent,
                   "%05.2f" % (beam.phase_noise_percent + 0.5 * const_sigma * beam.phase_noise_percent)]

    #print(levels_ticks_colorbar)
    #print(ticks_cbar)

    colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation="vertical", aspect=10, pad=0.05)
    colorbar.set_label("$\mathbf{\Delta \\varphi / 2\pi}$, %", labelpad=-120, y=1.15, rotation=0, fontsize=font_size, fontweight="bold")
    colorbar.ax.set_yticklabels(ticks_cbar)
    colorbar.ax.tick_params(labelsize=font_size - 10)

    plt.savefig(path + "/example_of_phase_screen.png", bbox_inches="tight")
    plt.close()

    del arr

def plot_track(states_arr, parameter_index, path):
    zs = [ e * 10**2 for e in states_arr[:,0] ]
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
