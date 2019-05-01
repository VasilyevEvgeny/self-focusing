from core.libs import *
from core.functions import create_dir


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


def plot_topological_charge(res_dir="misc/topological_charge/images", **kwargs):
    m = kwargs["m"]

    create_dir()

    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
        sleep(1)
    os.makedirs(res_dir)

    n_points = 100
    x_max, y_max = 600.0, 600.0  # micrometers

    x, y = np.zeros(n_points), np.zeros(n_points)
    for i in range(n_points):
        x[i], y[i] = i * x_max / n_points - x_max / 2, i * y_max / n_points - y_max / 2

    phase = np.zeros((n_points, n_points))
    phase = phase_initialization(phase, x, y, n_points, m)

    xx, yy = np.meshgrid(x, y)

    for i in tqdm(range(360)):
        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, phase, cmap="gray", rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.view_init(elev=75, azim=int(i + 315))
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_zlim([0, 2*pi+1])
        ax.xaxis.set_tick_params(pad=15)
        ax.yaxis.set_tick_params(pad=15)
        ax.zaxis.set_tick_params(pad=15)
        bbox = fig.bbox_inches.from_bounds(3, 2, 7.5, 6.1)
        plt.savefig(res_dir + '/%d.png' % i, bbox_inches=bbox, transparent=True)
        plt.close()


plot_topological_charge(m=3)
