from numba import jit
from numpy import exp, zeros, meshgrid
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core import create_dir, make_paths, make_animation, make_video, parse_args


@jit(nopython=True)
def intensity_initialization(n_points, x, y, x_0, y_0, M):
    intensity = zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            intensity[i, j] = ((x[i] / x_0) ** 2 + (y[j] / y_0) ** 2) ** M * \
                              exp(-((x[i] / x_0) ** 2 + (y[j] / y_0) ** 2))

    return intensity


def plot_images(**kwargs):
    global_root_dir = kwargs["global_root_dir"]
    global_results_dir_name = kwargs["global_results_dir_name"]
    prefix = kwargs["prefix"]
    M = kwargs["M"]
    figsize = kwargs.get("figsize", (10,10))
    ext = kwargs.get("ext", "png")

    _, results_dir, _ = make_paths(global_root_dir, global_results_dir_name, prefix)
    res_dir = create_dir(path=results_dir)

    n_points = 300
    x_max, y_max = 600.0, 600.0  # micrometers
    x_0, y_0 = x_max / 6, y_max / 6

    x, y = zeros(n_points), zeros(n_points)
    for i in range(n_points):
        x[i], y[i] = i * x_max / n_points - x_max / 2, i * y_max / n_points - y_max / 2

    intensity = intensity_initialization(n_points, x, y, x_0, y_0, M)

    xx, yy = meshgrid(x, y)

    for number, gradus in enumerate(tqdm(range(0, 360, 2))):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, intensity, cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.view_init(elev=75, azim=int(gradus + 315))
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_zlim([0, 1.35])

        if figsize == (3,3):
            bbox = fig.bbox_inches.from_bounds(0.6, 0.5, 2, 2)
        elif figsize == (10,10):
            bbox = fig.bbox_inches.from_bounds(2.3, 1.7, 5.7, 5.5)
        else:
            raise Exception("Wrong figsize!")
        plt.savefig(res_dir + '/%04d.' % number + ext, bbox_inches=bbox, transparent=True)
        plt.close()

    return results_dir


def process_initial(M, animation=True, video=True):
    args = parse_args()

    prefix = "M=%d" % M
    results_dir = plot_images(global_root_dir=args.global_root_dir,
                              global_results_dir_name=args.global_results_dir_name,
                              M=M,
                              prefix=prefix)

    if animation:
        make_animation(root_dir=results_dir,
                       name=prefix)

    if video:
        make_video(root_dir=results_dir,
                   name=prefix)


process_initial(M=1)
