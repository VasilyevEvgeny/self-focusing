from unittest import TestCase
import abc
from argparse import Namespace
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

from core import MathConstants, Medium, calculate_p_vortex, load_dirnames


class TestVortexCriticalPower(TestCase, metaclass=abc.ABCMeta):
    """
    Abstract class for testing the formula for critical self-focusing power of optical vortex.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global_root_dir, global_results_dir_name = load_dirnames()

        self._args = Namespace(global_root_dir=global_root_dir,
                               global_results_dir_name=global_results_dir_name)

        self._lmbda = randint(400, 3001) * 10**-9
        self._radius = randint(30, 80) * 10**-6
        self.__m_constants = MathConstants()
        self._medium = Medium(name=choice(['SiO2', 'CaF2', 'LiF']),
                              lmbda=self._lmbda,
                              m_constants=self.__m_constants)

        self._n_z = None
        self._n_z_diff = 10
        self._n_i_max_to_stop = 30

        self._eps = None
        self._png_name = None

        self._ms = [1, 2, 3, 4, 5]

        self._p_v_rel_true = array([calculate_p_vortex(m, 1) for m in self._ms])
        self._p_v_rel_pred = zeros(shape=self._p_v_rel_true.shape)

        self._p_vs = [0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08,
                      1.09, 1.1, 1.5, 2.0]

        self.__cmap = get_cmap('jet')
        self.__n_p_vs = len(self._p_vs)
        self._p_colors = [self.__cmap(i / (self.__n_p_vs-1)) for i in range(self.__n_p_vs)]

        self._flag_plot = True
        self._language = 'english'

    def _add_prefix(self, name):
        d = vars(self._args)
        d['prefix'] = 'test_' + name

    def _check(self):
        for i in range(len(self._p_v_rel_true)):
            p_rel_pred = self._p_v_rel_pred[i]
            self.assertLess(abs(p_rel_pred-1), self._eps)

    @abc.abstractmethod
    def process(self):
        """Numerical solution"""

    def _plot_propagation(self, dfs, path_to_save_plot, m):
        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15,10))
        for idx, (p_v_normalized, df) in enumerate(dfs):
            if p_v_normalized == 1.0:
                color = 'black'
                z_order = 1
                linewidth = 7
            else:
                color = self._p_colors[idx]
                linewidth = 3
                z_order = 0

            logarithmic = [ log10(e) for e in df['i_max_normalized'] ]
            plt.plot(df['z_normalized'], logarithmic, color=color, linewidth=linewidth, linestyle='solid',
                     alpha=0.8, label='$P_0/P_V = %2.2f$' % p_v_normalized, zorder=z_order)

        plt.xticks(fontsize=font_size-5, fontweight=font_weight)
        plt.yticks(fontsize=font_size-5, fontweight=font_weight)

        if self._language == 'english':
            xlabel = '$\mathbf{z \ / \ z_{diff}}$'
            ylabel = '$\mathbf{lg [ I_{max} \ / \ I_{max}(z=0)]}$'
        else:
            xlabel = '$\mathbf{z \ / \ z_{диф}}$'
            ylabel = '$\mathbf{lg [I_{макс} \ / \ I_{макс}(z=0)]}$'

        plt.xlabel(xlabel, fontsize=font_size, fontweight=font_weight)
        plt.ylabel(ylabel, fontsize=font_size, fontweight=font_weight)

        plt.ylim([-0.35, 1.1 * log10(self._n_i_max_to_stop)])

        plt.grid(linewidth=1, linestyle='dotted', alpha=0.5, color='gray')

        plt.legend(bbox_to_anchor=(0., 1.1, 1., .102), fontsize=font_size - 22, loc='center', ncol=4)

        plt.savefig(path_to_save_plot + '/i_max(z)_m=%d.png' % m, bbox_inches='tight')
        plt.close()

    def _plot_propagation_nice(self, dfs, path_to_save_plot, m):
        def cm2inch(*tupl):
            inch = 2.54
            if isinstance(tupl[0], tuple):
                return tuple(i / inch for i in tupl[0])
            else:
                return tuple(i / inch for i in tupl)

        plt.figure(figsize=cm2inch(11, 4))
        for idx, (p_v_normalized, df) in enumerate(dfs):
            if p_v_normalized == 1.0:
                color = 'black'
                z_order = 1
                linewidth = 3
            else:
                color = self._p_colors[idx]
                linewidth = 2
                z_order = 0

            logarithmic = [log10(e) for e in df['i_max_normalized']]
            plt.plot(df['z_normalized'], logarithmic, color=color, linewidth=linewidth, linestyle='solid',
                     alpha=0.8, label='$P_0/P_V^{(m)} = %2.2f$' % p_v_normalized, zorder=z_order)

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

        plt.savefig(path_to_save_plot + '/i_max(z)_m=%d.png' % m, bbox_inches='tight', dpi=500)
        plt.close()

    def _plot(self, path_to_save_plot, polyfit_degree=2):

        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15, 10))
        plt.scatter(self._ms, self._p_v_rel_pred, s=500, color='red')

        plt.axhline(1, linewidth=3, color='black', zorder=-1)

        # polynomial regression
        a, b, c = polyfit(self._ms, self._p_v_rel_pred, polyfit_degree)
        print('Curve: {}m^2 + {}m + {}'.format(a, b, c))
        xs = linspace(self._ms[0], self._ms[-1], 1000)
        ys = [a * x ** 2 + b * x + c for x in xs]
        sign_a = '+' if a >= 0 else '-'
        sign_b = '+' if b >= 0 else '-'
        sign_c = '+' if c >= 0 else '-'
        regr_label = '$%s$%05.3fm$^2$$%s$%05.3fm$%s$%05.3f' % (sign_a, abs(a), sign_b, abs(b), sign_c, abs(c))
        plt.plot(xs, ys, color='green', linewidth=10, alpha=0.5, label=regr_label)

        plt.ylim([0.8, 1.2])

        plt.xticks(self._ms, fontsize=font_size-5, fontweight=font_weight)
        plt.yticks(fontsize=font_size-5, fontweight=font_weight)

        plt.xlabel('$\mathbf{m}$', fontsize=font_size, fontweight=font_weight)

        if self._language == 'english':
            ylabel = '$\mathbf{P_V (numerical) \ / \ P_V (analytics)}$'
        else:
            ylabel = '$\mathbf{P_V (числ) \ / \ P_V (аналитич)}$'
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
        a, b, c = polyfit(self._ms, self._p_v_rel_pred, polyfit_degree)
        print('Curve: {}m^2 + {}m + {}'.format(a, b, c))
        xs = linspace(self._ms[0], self._ms[-1], 1000)
        ys = [a * x ** 2 + b * x + c for x in xs]
        sign_a = '+' if a >= 0 else '-'
        sign_b = '+' if b >= 0 else '-'
        sign_c = '+' if c >= 0 else '-'
        # regr_label = '$%s$%05.3fm$^2$$%s$%05.3fm$%s$%05.3f' % (sign_a, abs(a), sign_b, abs(b), sign_c, abs(c))
        errors = [abs(self._p_v_rel_pred[i] - 1) * 100 for i in range(len(self._ms))]

        fig = plt.figure(figsize=cm2inch(8, 8))
        ax = fig.add_subplot(111)

        #
        # errors
        #

        ax.fill_between(self._ms, errors, alpha=0.25, facecolor='blue', edgecolor=None,
                        label='$\\varepsilon$')


        x_ticks = self._ms
        x_ticklabels = ['{:d}'.format(e) for e in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontsize=10)
        ax.set_xlabel('$m$', fontsize=14)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 7)
        y_ticks = [0, 1, 2, 3, 4, 5, 6, 7]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
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
        ax_ghost.scatter(self._ms, self._p_v_rel_pred, s=50, color='red', zorder=0, label='численное решение')
        ax_ghost.plot(xs, ys, color='green', linewidth=2, zorder=1, label='квадратичная аппроксимация')

        dp = 0.01
        for i in range(len(self._ms)):
            ax_ghost.plot([self._ms[i], self._ms[i]], [self._p_v_rel_pred[i] - dp, self._p_v_rel_pred[i] + dp],
                          marker='_', c='black', lw=1)

        ax_ghost.set_ylim(0.96, 1.1)

        if self._language == 'english':
            ylabel = '$P_{V \mathrm{(num)}}^{(m)} \ / \ P_V^{(m)}$'
        else:
            ylabel = '$\mathbf{P_V (числ) \ / \ P_V (аналитич)}$'
        ax_ghost.set_ylabel(ylabel, fontsize=14)

        # ax_ghost.grid(c='gray', ls='-', lw=0.5, alpha=0.5)

        ax_ghost.legend(bbox_to_anchor=(-0.08, 1.15, 1., .102), handlelength=3.0,
                        fontsize=10, loc='center', ncol=1, frameon=False)

        plt.savefig(path_to_save_plot + '/' + self._png_name + '.png', bbox_inches='tight', dpi=500)
        plt.close()







        #
        # fig = plt.figure(figsize=cm2inch(8, 8))
        #
        # ax = fig.add_subplot(111)
        #
        # # errors
        # # ax.plot(self.__cores, self.__errors, c='blue', ls='-', lw=2, zorder=-1, alpha=0.25,
        # #         label='$\\varepsilon = (t_i^{theory} - t_i^{real}) / t_i^{theory} \\times 100$')
        # ax.fill_between(self.__cores, self.__errors, alpha=0.25, facecolor='blue', edgecolor=None,
        #                 label='$\\varepsilon$')
        #
        # x_ticks = [2, 4, 6, 8, 10, 12]
        # x_ticklabels = ['{:d}'.format(e) for e in x_ticks]
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_ticklabels, fontsize=10)
        # ax.set_xlabel('$i$', fontsize=18)
        #
        # ax.set_xlim(0, 13)
        # ax.set_ylim(-5, 90)
        # ax.set_ylabel('$\\varepsilon, \%$', fontsize=18)
        #
        # # plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        #
        # ax.grid(c='gray', ls=':', lw=0.5, alpha=0.5)
        #
        # ax.legend(bbox_to_anchor=(0.55, 1.05, 1., .102), handlelength=3.0,
        #           fontsize=10, loc='center', ncol=1, frameon=False)
        #
        # # ts
        #
        # ax_ghost = ax.twinx()
        # # ax.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=True, labelright=True,
        # #                bottom=False, labelbottom=False)
        # # ax_ghost.tick_params(top=False, labeltop=False, left=True, labelleft=True, right=False, labelright=False,
        # #                      bottom=False, labelbottom=False)
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position('right')
        # ax_ghost.yaxis.tick_left()
        # ax_ghost.yaxis.set_label_position('left')
        #
        # ax_ghost.scatter(self.__xs, self.__ys, c='black', marker='o', s=50, zorder=-1,
        #                  label='$\\overline{t}_i^{theory} = 1 / i$')
        # ax_ghost.scatter(self.__cores, self.__ts, c='red', marker='x', s=50, zorder=1,
        #                  label='$\\overline{t}_i^{real}$')
        #
        # # y_ticks = [i * 0.1 for i in range(11)]
        # # y_ticklabels = ['{:03.1f}'.format(e) for e in y_ticks]
        # # ax_ghost.set_yticks(y_ticks)
        # # ax_ghost.set_yticklabels(y_ticklabels, fontsize=10)
        #
        # ax_ghost.set_ylabel('$\\overline{t}_i$', fontsize=18)
        #
        # ax_ghost.grid(c='gray', ls='-', lw=0.5, alpha=0.5)
        #
        # ax_ghost.legend(bbox_to_anchor=(-0.25, 1.05, 1., .102), handlelength=3.0,
        #                 fontsize=10, loc='center', ncol=2, frameon=False)
        #
        # plt.savefig('threads.png', bbox_inches='tight', dpi=500)
        # plt.close()