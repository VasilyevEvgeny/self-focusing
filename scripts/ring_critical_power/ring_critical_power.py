import abc
from numpy.random import randint, choice
from numpy import array, zeros, log10, polyfit, linspace
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')

from core import MathConstants, Medium, parse_args


class RingCriticalPower(metaclass=abc.ABCMeta):
    """
    Abstract class for self-focusing study of ring beams without phase singularity.
    It is assumed that the class should be inherited from it, in which the method process is defined, where,
    in the presence or absence of an axisymmetric approximation, the search for a critical power of self-focusing for
    annular beam is realized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._args = parse_args()

        self._lmbda = randint(400, 3001) * 10**-9
        self._radius = randint(30, 80) * 10**-6
        self.__m_constants = MathConstants()
        self._medium = Medium(name=choice(['SiO2', 'CaF2', 'LiF']),
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self._n_z = None
        self._n_z_diff = 10
        self._n_i_max_to_stop = 10

        self._eps = None
        self._png_name = None

        self._Ms = array([1, 2, 3, 4, 5])

        self._p_g_rel_pred = zeros(shape=self._Ms.shape)

        self._p_gs = [ i * 0.02 + 1.0 for i in range(50) ]

        self.__cmap = get_cmap('jet')
        self.__n_p_gs = len(self._p_gs)
        self._p_colors = [self.__cmap(i / (self.__n_p_gs-1)) for i in range(self.__n_p_gs)]

        self._flag_plot = True
        self._language = 'english'

    def _add_prefix(self, name):
        d = vars(self._args)
        d['prefix'] = name

    @abc.abstractmethod
    def process(self):
        """Numerical solution"""

    def _plot_propagation(self, dfs, path_to_save_plot, M):
        """Plots beam peak intensity dependence on evolutionary coordinate z"""

        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15,10))
        for idx, (p_g_normalized, df) in enumerate(dfs):
            color = self._p_colors[idx]
            linewidth = 3
            logarithmic = [log10(e) for e in df['i_max_normalized']]
            plt.plot(df['z_normalized'], logarithmic, color=color, linewidth=linewidth, linestyle='solid',
                     alpha=0.8, label='$P_0/P_G = %2.2f$' % p_g_normalized)

        plt.xticks(fontsize=font_size-5, fontweight=font_weight)
        plt.yticks(fontsize=font_size-5, fontweight=font_weight)

        if self._language == 'english':
            xlabel = '$\mathbf{z \ / \ z_{diff}}$'
            ylabel = '$\mathbf{lg [I_{max} \ / \ I_{max}(z=0)]}$'
        else:
            xlabel = '$\mathbf{z \ / \ z_{диф}}$'
            ylabel = '$\mathbf{lg [I_{макс} \ / \ I_{макс}(z=0)]}$'

        plt.xlabel(xlabel, fontsize=font_size, fontweight=font_weight)
        plt.ylabel(ylabel, fontsize=font_size, fontweight=font_weight)

        plt.ylim([-0.35, 1.1 * log10(self._n_i_max_to_stop)])

        plt.grid(linewidth=1, linestyle='dotted', alpha=0.5, color='gray')

        plt.legend(bbox_to_anchor=(0., 1.1, 1., .4), fontsize=font_size - 22, loc='center', ncol=5)

        plt.savefig(path_to_save_plot + '/i_max(z)_M=%d.png' % M, bbox_inches='tight')
        plt.close()

    def _plot_propagation_nice(self, dfs, path_to_save_plot, M):
        def cm2inch(*tupl):
            inch = 2.54
            if isinstance(tupl[0], tuple):
                return tuple(i / inch for i in tupl[0])
            else:
                return tuple(i / inch for i in tupl)

        plt.figure(figsize=cm2inch(11, 4))
        for idx, (p_g, df) in enumerate(dfs):
            color = self._p_colors[idx]
            linewidth = 2
            z_order = 0

            logarithmic = [log10(e) for e in df['i_max_normalized']]
            plt.plot(df['z_normalized'], logarithmic, color=color, linewidth=linewidth, linestyle='solid',
                     alpha=0.8, label='$P_0/P_G = %2.2f$' % p_g, zorder=z_order)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        if self._language == 'english':
            xlabel = '$z \ / \ z_{\mathrm{diff}}$'
            ylabel = '$\lg(I_{\mathrm{max}} \ / \ I_{\mathrm{max} 0})$'
        else:
            xlabel = '$z \ / \ z_{\mathrm{диф}}$'
            ylabel = '$\lg(I_{\mathrm{макс}} \ / \ I_{\mathrm{макс} 0})$'

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        plt.ylim([-0.35, 1.1 * log10(self._n_i_max_to_stop)])

        plt.grid(linewidth=0.5, linestyle='dotted', alpha=0.5, color='gray')

        # plt.legend(bbox_to_anchor=(0., 1.5, 1., .102), fontsize=10, loc='center', ncol=4)

        plt.savefig(path_to_save_plot + '/i_max(z)_m=%d.png' % M, bbox_inches='tight', dpi=500)
        plt.close()

    def _plot(self, path_to_save_plot, polyfit_degree=1):
        """Plots numerically found critical power of self-focusing for the annular beam in unit of Gaussian critical
         power of self-focusing"""

        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15, 10))
        plt.scatter(self._Ms, self._p_g_rel_pred, s=500, color='red')

        # polynomial regression
        a, b = polyfit(self._Ms, self._p_g_rel_pred, polyfit_degree)
        xs = linspace(self._Ms[0], self._Ms[-1], 1000)
        ys = [a * x + b for x in xs]
        sign_a = '+' if a >= 0 else '-'
        sign_b = '+' if b >= 0 else '-'
        regr_label = '$%s$%05.3fM$%s$%05.3f' % (sign_a, abs(a), sign_b, abs(b))
        plt.plot(xs, ys, color='green', linewidth=10, alpha=0.5, label=regr_label)

        plt.xticks(self._Ms, fontsize=font_size-5, fontweight=font_weight)
        plt.yticks(fontsize=font_size-5, fontweight=font_weight)

        plt.xlabel('$\mathbf{M}$', fontsize=font_size, fontweight=font_weight)

        if self._language == 'english':
            ylabel = '$\mathbf{P_R (numerical) \ / \ P_G}$'
        else:
            ylabel = '$\mathbf{P_R (числ) \ / \ P_G}$'
        plt.ylabel(ylabel, fontsize=font_size, fontweight=font_weight)

        plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)
        plt.legend(fontsize=font_size)

        plt.savefig(path_to_save_plot + '/' + self._png_name + '.png', bbox_inches='tight')
        plt.close()

    def _plot_nice(self, path_to_save_plot, polyfit_degree=2):
        def cm2inch(*tupl):
            inch = 2.54
            if isinstance(tupl[0], tuple):
                return tuple(i / inch for i in tupl[0])
            else:
                return tuple(i / inch for i in tupl)

        # polynomial regression
        a, b, c = polyfit(self._Ms, self._p_g_rel_pred, polyfit_degree)
        print('Curve: {}m^2 + {}m + {}'.format(a, b, c))
        xs = linspace(self._Ms[0], self._Ms[-1], 1000)
        ys = [a * x ** 2 + b * x + c for x in xs]
        sign_a = '+' if a >= 0 else '-'
        sign_b = '+' if b >= 0 else '-'
        sign_c = '+' if c >= 0 else '-'
        # regr_label = '$%s$%05.3fm$^2$$%s$%05.3fm$%s$%05.3f' % (sign_a, abs(a), sign_b, abs(b), sign_c, abs(c))
        errors = [abs(self._p_g_rel_pred[i] - 1) * 100 for i in range(len(self._Ms))]

        fig = plt.figure(figsize=cm2inch(8, 8))
        ax = fig.add_subplot(111)

        #
        # errors
        #

        ax.fill_between(self._Ms, errors, alpha=0.25, facecolor='blue', edgecolor=None,
                        label='$\\varepsilon$')

        x_ticks = self._Ms
        x_ticklabels = ['{:d}'.format(e) for e in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontsize=10)
        ax.set_xlabel('$M$', fontsize=14)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.set_ylabel('$\\varepsilon, \%$', fontsize=14)

        # plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

        ax.grid(c='gray', ls=':', lw=0.5, alpha=0.5)

        ax.legend(bbox_to_anchor=(0.37, 1.06, 1., .102), handlelength=3.0,
                  fontsize=10, loc='center', ncol=1, frameon=False)

        #
        # p_v
        #

        ax_ghost = ax.twinx()
        # ax.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=True, labelright=True,
        #                bottom=False, labelbottom=False)
        # ax_ghost.tick_params(top=False, labeltop=False, left=True, labelleft=True, right=False, labelright=False,
        #                      bottom=False, labelbottom=False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax_ghost.yaxis.tick_left()
        ax_ghost.yaxis.set_label_position('left')

        ax_ghost.axhline(1, linewidth=2, color='black', zorder=-1, label='аналитическая формула')
        ax_ghost.scatter(self._Ms, self._p_g_rel_pred, s=50, color='red', zorder=0, label='численное решение')
        ax_ghost.plot(xs, ys, color='green', linewidth=2, zorder=1, label='квадратичная аппроксимация')

        ax_ghost.set_ylim(1.1, 1.8)

        if self._language == 'english':
            ylabel = '$P_{R \mathrm{(num)}}^{(M)} \ / \ P_G$'
        else:
            ylabel = '$\mathbf{P_V (числ) \ / \ P_V (аналитич)}$'
        ax_ghost.set_ylabel(ylabel, fontsize=14)

        ax_ghost.grid(c='gray', ls='-', lw=0.5, alpha=0.5)

        ax_ghost.legend(bbox_to_anchor=(-0.08, 1.15, 1., .102), handlelength=3.0,
                        fontsize=10, loc='center', ncol=1, frameon=False)

        plt.savefig(path_to_save_plot + '/' + self._png_name + '.png', bbox_inches='tight', dpi=500)
        plt.close()
