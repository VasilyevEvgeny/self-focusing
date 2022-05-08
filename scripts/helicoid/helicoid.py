import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core import create_dir, make_paths, parse_args


def plot_helicoid(**kwargs):
    global_root_dir = kwargs['global_root_dir']
    global_results_dir_name = kwargs['global_results_dir_name']
    prefix = kwargs['prefix']

    _, results_dir, _ = make_paths(global_root_dir, global_results_dir_name, prefix)
    res_dir = create_dir(path=results_dir)

    points = 100
    angle_min = 180
    angle_max = 700
    dangle = angle_max - angle_min
    r = 10.0
    c = 1.0
    n = dangle / 360

    u = np.linspace(0, r, endpoint=True, num=int(points * n))
    v = np.linspace(-np.deg2rad(angle_min), np.deg2rad(angle_max), endpoint=True, num=int(2 * points * n))
    u, v = np.meshgrid(u, v)

    r0 = 0.5 * r
    delta = 0.5 * r0
    amp = 2 * c
    x = u * np.cos(v) + amp * exp(-((u - r0) / delta)**2)
    y = u * np.sin(v) + amp * exp(-((u - r0) / delta)**2)
    z = c * v

    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(z, x, y, alpha=1, edgecolors='w', zorder=0, color='black')
    ax.quiver(-6, 0, 0, 24, 0, 0, length=1, arrow_length_ratio=0.03, color='black', zorder=1)
    ax.text(19, -2, -2, "$\mathbf{z}$", fontsize=40, color='black', zorder=1)
    ax._axis3don = False
    bbox = fig.bbox_inches.from_bounds(8, 2, 17, 6)
    plt.savefig(res_dir + '/helicoid.png', bbox_inches=bbox)


args = parse_args()
plot_helicoid(global_root_dir=args.global_root_dir,
              global_results_dir_name=args.global_results_dir_name,
              prefix=args.prefix)
