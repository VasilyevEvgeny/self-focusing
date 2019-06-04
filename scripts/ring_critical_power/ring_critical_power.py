import abc
from numpy.random import randint, choice
from numpy import array, zeros
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from core import M_Constants, Medium, parse_args


class RingCriticalPower(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._args = parse_args()

        self._lmbda = randint(400, 3001) * 10**-9
        self._radius = randint(30, 151) * 10**-6
        self.__m_constants = M_Constants()
        self._medium = Medium(name=choice(['SiO2', 'CaF2', 'LiF']),
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self._n_z = None
        self._n_z_diff = 5
        self._n_i_max_to_stop = 10

        self._eps = None
        self._png_name = None

        self._Ms = array([5, 4, 3, 2, 1])

        self._p_g_rel_pred = zeros(shape=self._Ms.shape)

        self._p_gs = [i * 0.01 for i in range(1, 10)] + [i * 0.1 for i in range(1, 16)]

        self.__cmap = get_cmap('jet')
        self.__n_p_gs = len(self._p_gs)
        self._p_colors = [self.__cmap(i / (self.__n_p_gs-1)) for i in range(self.__n_p_gs)]

        self._flag_plot = True
        self._language = 'english'

    def add_prefix(self, name):
        d = vars(self._args)
        d['prefix'] = 'test_' + name

    @abc.abstractmethod
    def process(self):
        """Numerical solution"""

    def plot_propagation(self, dfs, path_to_save_plot, m):
        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15,10))
        for idx, (p_g_normalized, df) in enumerate(dfs):
            color = self._p_colors[idx]
            linewidth = 3
            plt.plot(df['z_normalized'], df['i_max_normalized'], color=color, linewidth=linewidth, linestyle='solid',
                     alpha=0.8, label='$P_0/P_G = %2.2f$' % p_g_normalized)

        plt.xticks(fontsize=font_size-5, fontweight=font_weight)
        plt.yticks(fontsize=font_size-5, fontweight=font_weight)

        if self._language == 'english':
            xlabel = '$\mathbf{z \ / \ z_{diff}}$'
            ylabel = '$\mathbf{I_{max} \ / \ I_{max}(z=0)}$'
        else:
            xlabel = '$\mathbf{z \ / \ z_{диф}}$'
            ylabel = '$\mathbf{I_{макс} \ / \ I_{макс}(z=0)}$'

        plt.xlabel(xlabel, fontsize=font_size, fontweight=font_weight)
        plt.ylabel(ylabel, fontsize=font_size, fontweight=font_weight)

        plt.grid(linewidth=1, linestyle='dotted', alpha=0.5, color='gray')

        plt.legend(bbox_to_anchor=(0., 1.1, 1., .202), fontsize=font_size - 22, loc='center', ncol=4)

        plt.savefig(path_to_save_plot + '/i_max(z)_M=%d.png' % m, bbox_inches='tight')
        plt.close()

    def plot(self, path_to_save_plot):

        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15, 10))
        plt.scatter(self._Ms, self._p_g_rel_pred, s=500, color='red')

        plt.xticks(self._Ms, fontsize=font_size-5, fontweight=font_weight)
        plt.yticks(fontsize=font_size-5, fontweight=font_weight)

        plt.ylim([0, 1.5])

        plt.xlabel('$\mathbf{m}$', fontsize=font_size, fontweight=font_weight)

        if self._language == 'english':
            ylabel = '$\mathbf{P_R (numerical) \ / \ P_G}$'
        else:
            ylabel = '$\mathbf{P_R (числ) \ / \ P_G}$'
        plt.ylabel(ylabel, fontsize=font_size, fontweight=font_weight)

        plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)

        plt.savefig(path_to_save_plot + '/' + self._png_name + '.png', bbox_inches='tight')
        plt.close()
