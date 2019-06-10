from unittest import TestCase
import abc
from matplotlib import pyplot as plt
from numpy import sqrt, arctan, exp
from numpy.linalg import norm
from numpy import max as maximum
from numpy.random import randint, choice
from argparse import Namespace

from core import Medium, MathConstants, load_dirnames


class TestDiffraction(TestCase, metaclass=abc.ABCMeta):
    """
    Abstract class for testing diffraction of different beams.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global_root_dir, global_results_dir_name = load_dirnames()

        self._args = Namespace(global_root_dir=global_root_dir,
                               global_results_dir_name=global_results_dir_name,
                               insert_datetime=True)

        self._lmbda = randint(400, 3001) * 10**-9
        self._radius = randint(30, 151) * 10**-6
        self.__m_constants = MathConstants()
        self._medium = Medium(name=choice(['SiO2', 'CaF2', 'LiF']),
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self._n_z = randint(100, 1001)

        self._p_0_to_p_gauss = randint(1, 11)
        self._p_0_to_p_vortex = randint(1, 11)

        self._p = None
        self._eps = None
        self._png_name = None

        self._flag_plot = True
        self._language = 'english'
        self._horizontal_lines = {'$1 \ / \ /sqrt{2}$': 1 / sqrt(2),
                                  '$1 \ / \ 2$': 1 / 2}
        self._horizontal_line = None

    def _add_prefix(self, name):
        d = vars(self._args)
        d['prefix'] = 'test_' + name

    def __calculate_max_intensity(self, z, spatial_coord=0.0):
        z_rel = z / (self._medium.k_0 * self._radius**2)
        a_0 = exp(-spatial_coord**2 / (self._radius**2 * (1 + z_rel**2))) / sqrt(1.0 + z_rel**2) ** self._p
        k_psi = (spatial_coord / self._radius)**2 * z_rel / (1 + z_rel**2) ** self._p - arctan(z_rel)

        return norm(a_0 * exp(1j * k_psi))**2

    def _check(self, df):
        self.assertLess(maximum(abs(df['i_max / i_0'] - df['analytics']) / df['analytics']), self._eps)

    @abc.abstractmethod
    def process(self):
        """Numerical solution"""

    def _plot(self, df, path_to_save_plot, z_diff):
        df['z_normalized'] /= z_diff

        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15, 10))
        if self._language == 'english':
            label_numerical = 'Numerical\nmodeling'
            label_analytics = 'Analytics'
        else:
            label_numerical = 'Численное\nмоделирование'
            label_analytics = 'Аналитическая\nформула'
        plt.plot(df['z_normalized'], df['analytics'], color='blue', linewidth=30, label=label_analytics)
        plt.plot(df['z_normalized'], df['i_max / i_0'], color='red', linewidth=10, label=label_numerical)
        plt.xticks(fontsize=font_size-5)
        plt.yticks(fontsize=font_size-5)

        if self._language == 'english':
            x_label = '$\mathbf{z \ / \ z_{diff}}$'
        else:
            x_label = '$\mathbf{z \ / \ z_{диф}}$'
        plt.xlabel(x_label, fontsize=font_size, fontweight=font_weight)
        plt.ylabel('$\mathbf{I \ / \ I_0}$', fontsize=font_size, fontweight=font_weight)

        if self._horizontal_line == 1 / sqrt(2):
            plt.axhline(1 / sqrt(2), color='black', linestyle='solid', linewidth=5, zorder=-1)
            plt.text(0.05, 0.72, "$1 / \sqrt{2}$", fontsize=font_size)
        elif self._horizontal_line == 1 / 2:
            plt.axhline(1 / 2, color='black', linestyle='solid', linewidth=5, zorder=-1)
            plt.text(0.05, 0.53, "$1 / 2$", fontsize=font_size)

        plt.grid(linewidth=2, color='gray', linestyle='dotted', alpha=0.5)
        plt.legend(fontsize=font_size-10, loc=1)

        plt.savefig(path_to_save_plot + '/' + self._png_name + '.png', bbox_inches='tight')
        plt.close()

    def _add_analytics_to_df(self, df):
        df['analytics'] = 0.0
        n = len(df)
        for i in range(n):
            z = df['z_normalized'][i]
            df['analytics'][i] = self.__calculate_max_intensity(z)
