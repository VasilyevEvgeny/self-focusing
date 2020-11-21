from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
from numpy import zeros, complex64, float64, exp, arctan2, pi, sqrt, log10, where
from numpy.fft import fft2, fftshift
from numba import jit


class InitialSpectrum:
    def __init__(self, **kwargs):
        self.__n_perp = kwargs['n_perp']
        self.__perp_max = kwargs['perp_max']
        self.__d_perp = self.__perp_max / self.__n_perp

        self.__m = kwargs['m']
        self.__r_0 = kwargs['r_0']

        # self.__kerr_coeff = kwargs['kerr_coeff']

        self.__arr = zeros((self.__n_perp, self.__n_perp), dtype=complex64)
        self.__arr_norm = zeros((self.__n_perp, self.__n_perp), dtype=float64)
        self.__arr_norm_cropped = None

        self.__spectrum = zeros((self.__n_perp, self.__n_perp), dtype=complex64)
        self.__spectrum_norm = zeros((self.__n_perp, self.__n_perp), dtype=float64)
        self.__spectrum_norm_cropped = None

    @staticmethod
    @jit(nopython=True)
    def __initialize_arr(arr, perp_max, d_perp, r_0, m):
        r1 = 100 * 10**-6
        r2 = 400 * 10**-6
        d1 = 30 * 10 ** -6
        d2 = 50 * 10 ** -6

        N, M = arr.shape[0], arr.shape[1]
        for i in range(N):
            for j in range(M):
                x, y = d_perp * i - 0.5 * perp_max, d_perp * j - 0.5 * perp_max
                r = sqrt(x**2 + y**2)

                # ring width influence on spectrum
                r1 = 100 * 10 ** -6
                d1 = 10 * 10 ** -6  # (10 and 30)
                arr[i, j] = exp(-0.5 * ((r - r1) ** 2) / d1 ** 2)
                # arr[i, j] *= exp(1j * m * (arctan2(x, y)))



                # arr[i, j] = exp(-0.5 * ((r - r1) ** 2) / d1 ** 2)  # + 0.19 * exp(-0.5 * ((r - r2) ** 2) / d2 ** 2)

                # if r < r1 + 0.5 * (r2 - r1):
                #     arr[i, j] *= exp(1j * m * (arctan2(x, y)))
                # else:
                #     arr[i, j] *= exp(1j * m * (arctan2(x, y) + pi / m))

                # arr[i, j] = (r / r_0)**abs(m) * exp(-0.5 * (r / r_0)**2) * \
                #             exp(1j * m * (arctan2(x, y) + pi))

    @staticmethod
    @jit(nopython=True)
    def __normalize(arr):
        N, M = arr.shape[0], arr.shape[1]
        arr_norm = zeros((N, M), dtype=float64)
        for i in range(N):
            for j in range(M):
                arr_norm[i, j] = arr[i, j].real**2 + arr[i, j].imag**2

        return arr_norm

    def __make_fft(self):
        self.__spectrum = fft2(self.__arr)
        self.__spectrum = fftshift(self.__spectrum, axes=(0, 1))

    def __crop_arr(self, arr, remaining_central_part_coeff):
        """
        :param remaining_central_part_coeff: 0 -> no points, 0.5 -> central half number of points, 1.0 -> all points
        :return:
        """

        if remaining_central_part_coeff < 0 or remaining_central_part_coeff > 1:
            raise Exception('Wrong remaining_central part_coeff!')

        N = arr.shape[0]
        delta = int(remaining_central_part_coeff / 2 * N)
        i_min, i_max = N // 2 - delta, N // 2 + delta

        return arr[i_min:i_max, i_min:i_max]

    @staticmethod
    def __log_arr(arr):
        return where(arr < 0.1, -1, log10(arr))

    @staticmethod
    def __log_spectrum(arr):
        return log10(arr / np.max(arr))

    def __plot(self):

        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, hspace=1)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        # ax1.set_title('$\mathbf{I(x, y)}$')
        # ax2.set_title('$\mathbf{S(k_x, k_y)}$')

        #
        # ring influence on spectrum
        #
        arr_for_plot = self.__crop_arr(self.__arr_norm, 0.2)
        spectrum_for_plot = self.__crop_arr(self.__spectrum_norm, 0.02)
        # ax1
        ax1_xticks = [100, 205, 310]
        ax1.set_xticks(ax1_xticks)
        ax1.set_xticklabels(['$\mathbf{-\\xi}$', '0', '$\mathbf{+\\xi}$'], fontsize=40, fontweight='bold')
        ax1.grid(color='white', lw=3, ls=':', alpha=0.5)
        ax1.set_yticks(ax1_xticks)
        ax1.set_yticklabels(['$\mathbf{+\\xi}$', '0', '$\mathbf{-\\xi}$'], fontsize=40, fontweight='bold')
        ax1.grid(color='white', lw=3, ls=':', alpha=0.5)
        # ax2
        ax2_xticks = [13, 20, 27]
        ax2.set_xticks(ax2_xticks)
        ax2.set_xticklabels(['$\mathbf{-k_{\\xi}}$', '0', '$\mathbf{+k_{\\xi}}$'], fontsize=40, fontweight='bold')
        ax2.grid(color='white', lw=3, ls=':', alpha=0.5)
        ax2.set_yticks(ax2_xticks)
        ax2.set_yticklabels(['$\mathbf{+k_{\\xi}}$', '0', '$\mathbf{-k_{\\xi}}$'], fontsize=40, fontweight='bold')
        ax2.grid(color='white', lw=3, ls=':', alpha=0.5)

        ax1.contourf(arr_for_plot, cmap=cm.jet, levels=100)
        ax2.contourf(spectrum_for_plot, cmap=cm.gray, levels=100)

        plt.savefig('scripts/angular_spectrum/initial_spectrum/fft_vortex.png')
        # bbox_inches='tight'
        plt.show()
        plt.close()

    def process(self):
        self.__initialize_arr(self.__arr, self.__perp_max, self.__d_perp, self.__r_0, self.__m)
        self.__arr_norm = self.__normalize(self.__arr)

        self.__make_fft()
        self.__spectrum_norm = self.__normalize(self.__spectrum)

        self.__plot()


initial_vortex = InitialSpectrum(n_perp=2048,
                                 perp_max=2000 * 10**-6,
                                 m=1,
                                 r_0=15 * 10**-6)

initial_vortex.process()
