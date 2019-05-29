from unittest import TestCase
import abc
from argparse import Namespace
from numpy.random import randint, choice
from numpy import sqrt, array, zeros
from numpy import max as maximum
from matplotlib import pyplot as plt

from core import M_Constants, Medium


class TestMarburger(TestCase, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._args = Namespace(global_root_dir='L:/Vasilyev',
                               global_results_dir_name='Self-focusing_results')

        self._lmbda = randint(400, 3001) * 10**-9
        self._radius = randint(30, 151) * 10**-6
        self.__m_constants = M_Constants()
        self._medium = Medium(name=choice(['SiO2', 'CaF2', 'LiF']),
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self._n_z = 10000

        self._eps = None
        self._png_name = None

        self._p_rel_min = 2
        self._p_rel_max = 10
        self._p_rels = [p_rel for p_rel in range(self._p_rel_min, self._p_rel_max + 1, 1)]

        self._z_fil_rel_true = self.calculate_z_fil_rel_true()
        self._z_fil_rel_pred = zeros(shape=self._z_fil_rel_true.shape)

        self._flag_plot = True
        self._language = 'english'

    def add_prefix(self, name):
        d = vars(self._args)
        d['prefix'] = 'test_' + name

    def calculate_z_fil_rel_true(self):
        z_fil_rel_trues = []
        for p_rel in range(self._p_rel_min, self._p_rel_max+1, 1):
            z_fil_rel_trues.append(0.367 / sqrt((sqrt(p_rel) - 0.852)**2 - 0.0219))

        return array(z_fil_rel_trues)

    def check(self):
        for i in range(len(self._p_rels)):
            self.assertLess(abs(self._z_fil_rel_pred[i] - self._z_fil_rel_true[i]) / self._z_fil_rel_true[i], self._eps)

    @abc.abstractmethod
    def process(self):
        """Numerical solution"""

    def plot(self, path_to_save_plot):
        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15, 10))
        if self._language == 'english':
            label_numerical = 'Numerical\nmodeling'
            label_analytics = 'Analytics'
        else:
            label_numerical = 'Численное\nмоделирование'
            label_analytics = 'Аналитическая\nформула'
        plt.plot(self._p_rels, self._z_fil_rel_true, color='blue', linewidth=30, label=label_analytics)
        plt.scatter(self._p_rels, self._z_fil_rel_pred, color='red', linewidth=10, label=label_numerical)
        plt.xticks(fontsize=font_size-5)
        plt.yticks(fontsize=font_size-5)

        plt.xlabel('$\mathbf{P \ / \ P_{cr}}$', fontsize=font_size, fontweight=font_weight)
        if self._language == 'english':
            y_label = '$\mathbf{z \ / \ z_{diff}}$'
        else:
            y_label = '$\mathbf{z \ / \ z_{диф}}$'
        plt.ylabel(y_label, fontsize=font_size, fontweight=font_weight)

        plt.grid(linewidth=2, color='gray', linestyle='dotted', alpha=0.5)
        plt.legend(fontsize=font_size-10, loc=1)

        plt.savefig(path_to_save_plot + '/' + self._png_name + '.png', bbox_inches='tight')
        plt.close()

