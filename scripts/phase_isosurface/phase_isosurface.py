from glob import glob
from os import path
from numpy import load, array, float64, complex64, angle, pi
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

from core import create_dir, make_paths, parse_args


class PhaseSurface:
    def __init__(self, **kwargs):
        global_root_dir = kwargs['global_root_dir']
        global_results_dir_name = kwargs['global_results_dir_name']
        prefix = kwargs['prefix']

        _, results_dir, _ = make_paths(global_root_dir, global_results_dir_name, prefix)
        self.__res_dir = create_dir(path=results_dir)

        self.__phase = self.__load_field(kwargs['path'])
        self.__plot()

    @staticmethod
    def __load_field(path_to_npy):
        field = []
        for file in glob(path.join(path_to_npy, '*.npy')):
            field.append(load(file))

        return array(angle(array(field, dtype=complex64)) + pi, dtype=float64)

    def __plot(self):

        for azim in list(range(0, 360, 10)):
            fig = plt.figure(figsize=(15, 10))
            ax = fig.gca(projection='3d')

            levels = [2 * pi - 0.01]
            n = len(levels)
            alphas = [0.5]
            cmap = cm.get_cmap('Greys')
            colors = [cmap(i / n) for i in range(0, n)]
            for i, level in enumerate(levels):
                verts, faces, _, _ = measure.marching_cubes_lewiner(self.__phase, level, spacing=(0.1, 0.1, 0.1))
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                color=colors[i],
                                lw=1, alpha=alphas[i], zorder=1)
            ax.view_init(elev=25, azim=azim)  # 290
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            # ax.set_axis_off()

            plt.savefig(self.__res_dir + '/phase_isosurface_%03d.png' % azim, bbox_inches='tight', transparent=False)
            plt.close()


args = parse_args()
phase_surface = PhaseSurface(global_root_dir=args.global_root_dir,
                             global_results_dir_name=args.global_results_dir_name,
                             prefix=args.prefix,
                             path='R:/Self-focusing/spectrum_xy_2020-03-24_23-27-32/beam/field')
