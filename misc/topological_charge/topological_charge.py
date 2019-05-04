from core.libs import *
from core.functions import create_dir, make_paths, make_animation, make_video
from core.args import parse_args


@jit(nopython=True)
def calc_phase(arg):
    while arg >= 2 * pi:
        arg -= 2 * pi

    return arg


@jit(nopython=True)
def phase_initialization(phase, x, y, n_points, m):
    for i in range(n_points):
        for j in range(n_points):
            phase[i, j] = calc_phase(m * arctan2(y[i], x[j]) + m * pi)

    return phase


def plot_images(**kwargs):
    global_root_dir = kwargs["global_root_dir"]
    global_results_dir_name = kwargs["global_results_dir_name"]
    prefix = kwargs["prefix"]
    m = kwargs["m"]

    _, results_dir, _ = make_paths(global_root_dir, global_results_dir_name, prefix)
    res_dir = create_dir(path=results_dir)

    n_points = 400
    x_max, y_max = 600.0, 600.0  # micrometers

    x, y = np.zeros(n_points), np.zeros(n_points)
    for i in range(n_points):
        x[i], y[i] = i * x_max / n_points - x_max / 2, i * y_max / n_points - y_max / 2

    phase = np.zeros((n_points, n_points))
    phase = phase_initialization(phase, x, y, n_points, m)

    xx, yy = np.meshgrid(x, y)

    for number, gradus in enumerate(tqdm(range(0, 360, 2))):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, phase, cmap="gray", rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.view_init(elev=75, azim=int(gradus + 315))
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_zlim([0, 2*pi+1])
        ax.xaxis.set_tick_params(pad=15)
        ax.yaxis.set_tick_params(pad=15)
        ax.zaxis.set_tick_params(pad=15)
        #bbox = fig.bbox_inches.from_bounds(3, 2, 7.5, 6.1)
        bbox = fig.bbox_inches.from_bounds(0.6, 0.5, 2, 2)
        plt.savefig(res_dir + '/%04d.png' % number, bbox_inches=bbox, transparent=True)
        plt.close()

    return results_dir


def process_topological_charge(m, animation=True, video=True):
    args = parse_args()

    prefix = "m=%d" % m
    results_dir = plot_images(global_root_dir=args.global_root_dir,
                              global_results_dir_name=args.global_results_dir_name,
                              m=m,
                              prefix=prefix + "_")

    if animation:
        make_animation(root_dir=results_dir,
                       name=prefix)

    if video:
        make_video(root_dir=results_dir,
                   name=prefix)


process_topological_charge(m=3)
