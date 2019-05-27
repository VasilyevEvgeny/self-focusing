from argparse import Namespace
import unittest
from numpy import sqrt
from matplotlib import pyplot as plt

from core import BeamX, Propagator, SweepDiffractionExecutorX, xlsx_to_df
from tests.diffraction.analytics import Diffraction2d


args = Namespace(global_root_dir='L:/Vasilyev',
                 global_results_dir_name='Self-focusing_results',
                 prefix='test_diffraction_2d_gauss')


class TestDiffractionGauss2d(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDiffractionGauss2d, self).__init__(*args, **kwargs)

        self.__medium = 'SiO2'
        self.__lmbda = 1800 * 10**-9
        self.__x_0 = 100 * 10 ** -6

        self.__diffraction_2d_analytics = Diffraction2d(lmbda=self.__lmbda,
                                                        medium=self.__medium,
                                                        x_0=self.__x_0)

        self.__flag_plot = True
        self.__language = 'english'

    def check(self, df):
        n = len(df)
        for i in range(n):
            self.assertAlmostEqual(df['i_max / i_0'][i], df['analytics'][i], places=2)

    def process(self):
        beam = BeamX(medium=self.__medium,
                     M=0,
                     half=False,
                     lmbda=1800 * 10 ** -9,
                     x_0=self.__x_0,
                     n_x=512,
                     r_kerr=75.40)

        propagator = Propagator(args=args,
                                beam=beam,
                                diffraction=SweepDiffractionExecutorX(beam=beam),
                                n_z=1000,
                                dz0=beam.z_diff / 1000,
                                flag_const_dz=True,
                                dn_print_current_state=0,
                                dn_plot_beam=0,
                                beam_normalization_type='local')

        propagator.propagate()

        return propagator.logger.track_filename, propagator.manager.results_dir, propagator.beam.z_diff

    def plot(self, df, path_to_save_plot, z_diff):
        df['z_normalized'] /= z_diff

        font_size = 40
        font_weight = 'bold'
        plt.figure(figsize=(15, 10))
        if self.__language == 'english':
            label_numerical = 'Numerical\nmodeling'
            label_analytics = 'Analytics'
        else:
            label_numerical = 'Численное\nмоделирование'
            label_analytics = 'Аналитическая\nформула'
        plt.plot(df['z_normalized'], df['analytics'], color='blue', linewidth=30, label=label_analytics)
        plt.plot(df['z_normalized'], df['i_max / i_0'], color='red', linewidth=10, label=label_numerical)
        plt.xticks(fontsize=font_size-5)
        plt.yticks(fontsize=font_size-5)

        if self.__language == 'english':
            xlabel = '$\mathbf{z \ / \ z_{diff}}$, cm'
        else:
            xlabel = '$\mathbf{z \ / \ z_{diff}}$, см'
        plt.xlabel(xlabel, fontsize=font_size, fontweight=font_weight)
        plt.ylabel('$\mathbf{I \ / \ I_0}$', fontsize=font_size, fontweight=font_weight)

        plt.axhline(1 / sqrt(2), color='black', linestyle='solid', linewidth=5, zorder=-1)

        plt.grid(linewidth=2, color='gray', linestyle='dotted', alpha=0.5)

        plt.legend(fontsize=font_size-10, loc=1)
        plt.text(0.05, 0.72, "$1 / \sqrt{2}$", fontsize=font_size)

        plt.savefig(path_to_save_plot + '/diffraction_gauss_2d.png', bbox_inches='tight')
        plt.close()

    def add_analytics_to_df(self, df):
        df['analytics'] = 0.0
        n = len(df)
        for i in range(n):
            z = df['z_normalized'][i]
            df['analytics'][i] = self.__diffraction_2d_analytics.calculate_max_intensity(z)

    def test_diffraction_gauss_2d(self):
        track_filename, path_to_save_plot, z_diff = self.process()
        df = xlsx_to_df(track_filename, normalize_z_to=1)

        self.add_analytics_to_df(df)
        self.check(df)

        if self.__flag_plot:
            self.plot(df, path_to_save_plot, z_diff)

