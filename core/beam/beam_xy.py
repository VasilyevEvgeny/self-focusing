from numpy import pi, arctan2, exp, sqrt, zeros, complex64, mean, sum as summ, array, save
from scipy.special import gamma
from numba import jit

from .beam_3d import Beam3D


class BeamXY(Beam3D):
    """
    Subsubclass for 3-dimensional beam with spatial coordinates x and y
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__x_0 = kwargs['x_0']  # characteristic spatial size along x
        self.__y_0 = kwargs['y_0']  # characteristic spatial size along y

        self.__x_max = self._radii_in_grid * max(self.__x_0, self.__y_0)  # spatial grid size along x
        self.__y_max = self.__x_max  # spatial grid size along y

        self.__n_x = kwargs['n_x']  # number of points in spatial grid along x
        self.__n_y = kwargs['n_y']  # number of points in spatial grid along y

        self.__dx = self.__x_max / self.__n_x  # spatial grid step along x
        self.__dy = self.__y_max / self.__n_y  # spatial grid step along y

        self.__xs = [i * self.__dx - 0.5 * self.__x_max for i in range(self.__n_x)]  # spatial grid nodes along x
        self.__ys = [i * self.__dy - 0.5 * self.__y_max for i in range(self.__n_y)]  # spatial grid nodes along y

        self.__dk_x = 2.0 * pi / self.__x_max  # wave vector step along x
        self.__dk_y = 2.0 * pi / self.__y_max  # wave vector step along y

        self.__k_xs = array([i * self.__dk_x if i < self.__n_x / 2 else (i - self.__n_x) * self.__dk_x  # wave vector grid
                       for i in range(self.__n_x)])                                                     # nodes along x

        self.__k_ys = array([i * self.__dk_y if i < self.__n_y / 2 else (i - self.__n_y) * self.__dk_y  # wave vector grid
                       for i in range(self.__n_y)])                                                     # nodes along y

        self.__noise_percent = kwargs.get('noise_percent', 0.0)  # multiplicative noise percent
        self.__noise_field = zeros(shape=(self.__n_x, self.__n_y))  # array for complex noise field

        # noise initialization
        if self.__noise_percent:
            self.__noise = kwargs['noise']
            self.__noise.initialize(n_x=self.__n_x,
                                    n_y=self.__n_y,
                                    dx=self.__dx,
                                    dy=self.__dy)
            self.__noise.process()
            self.__noise_field = self.__noise.noise_field

        # field initialization
        self._field = self.__initialize_field(self._M, self._m, self.__x_0, self.__y_0, self.__x_max, self.__y_max,
                                              self.__dx, self.__dy, self.__n_x, self.__n_y, self.__noise_percent,
                                              self.__noise_field)

        # other parameters initialization
        self._i_0 = self.__calculate_i_0()
        self._z_diff = self._medium.k_0 * mean([self.__x_0, self.__y_0])**2
        self._r_kerr = 2 * self.medium.k_0 * self.medium.n_2 * self._i_0 * self._z_diff / self.medium.n_0

        self.update_intensity()

    @property
    def info(self):
        return 'beam_xy'

    @property
    def x_0(self):
        return self.__x_0

    @property
    def y_0(self):
        return self.__x_0

    @property
    def x_max(self):
        return self.__x_max

    @property
    def y_max(self):
        return self.__y_max

    @property
    def n_x(self):
        return self.__n_x

    @property
    def n_y(self):
        return self.__n_y

    @property
    def dx(self):
        return self.__dx

    @property
    def dy(self):
        return self.__dy

    @property
    def xs(self):
        return self.__xs

    @property
    def ys(self):
        return self.__ys

    @property
    def k_xs(self):
        return self.__k_xs

    @property
    def k_ys(self):
        return self.__k_ys

    @property
    def noise_percent(self):
        return self.__noise_percent

    @property
    def i_0(self):
        return self._i_0

    @property
    def z_diff(self):
        return self._z_diff

    @property
    def noise(self):
        return self.__noise

    @staticmethod
    @jit(nopython=True)
    def __calculate_intensity_intergral(intensity, dx, dy):
        """
        LATEX SYNTAX:
        intensity_integral = \iint\limits_{-\infty}^{+\infty} i(x, y) dx dy

        :param intensity: intensity array
        :param dx: spatial grid step along x
        :param dy: spatial grid step along y

        :return:
        """
        intensity_intergral = summ(intensity) * dx * dy

        return intensity_intergral

    def __calculate_i_0(self):
        """
        LATEX SYNTAX:
        P_0 = \int\limits_0^{+\infty} I_0(r) 2 \pi r dr = I_0 \int\limits_0^{+\infty} i(r) 2 \pi r dr = const I_0
        -->
        I_0 = P_0 / const

        :return: I_0
        """
        if self.__noise_percent == 0.0 and self.__x_0 == self.__y_0:
            return self._p_0 / (pi * self.__x_0**2 * gamma(self._M+1))
        else:
            return self._p_0 / self.__calculate_intensity_intergral(self._field_to_intensity(self._field),
                                                                    self.__dx, self.__dy)

    @staticmethod
    @jit(nopython=False)
    def __initialize_field(M, m, x_0, y_0, x_max, y_max, dx, dy, n_x, n_y, noise_percent, noise):
        """
        :param M: power of polynomial before exponent in initial condition
        :param m: topological charge
        :param x_0: characteristic spatial size along x
        :param y_0: characteristic spatial size along y
        :param x_max: spatial grid size along x
        :param y_max: spatial grid size along y
        :param dx: spatial grid step along x
        :param dy: spatial grid step along y
        :param n_x: number of points in spatial grid along x
        :param n_y: number of points in spatial grid along y
        :param noise_percent: multiplicative noise percent
        :param noise: array for complex noise field

        :return: initialized field array
        """
        arr = zeros(shape=(n_x, n_y), dtype=complex64)
        for i in range(n_x):
            for j in range(n_y):
                x, y = i * dx - 0.5 * x_max, j * dy - 0.5 * y_max

                #
                # nested vortices
                #

                r1 = 100 * 10 ** -6
                d1 = 10 * 10 ** -6

                r2 = 300 * 10 ** -6
                d2 = 50 * 10 ** -6

                r = sqrt(x ** 2 + y ** 2)

                arr[i, j] = exp(-0.5 * ((r - r1) ** 2) / d1 ** 2) + 0.4 * exp(-0.5 * ((r - r2) ** 2) / d2 ** 2)

                if r < 0.5 * (r1 + r2):
                    arr[i, j] *= exp(1j * m * (arctan2(x, y)))
                else:
                    arr[i, j] *= exp(1j * 1 * (arctan2(x, y) + pi))

                # #
                # # ring width
                # #
                #
                # r0 = 300 * 10 ** -6
                # d = 50 * 10 ** -6
                # r = sqrt(x ** 2 + y ** 2)
                # arr[i, j] = exp(-0.5 * ((r - r0) ** 2) / d ** 2)


                #  OLD
                # arr[i, j] = (1.0 + 0.01 * noise_percent * noise[i, j]) * \
                #             sqrt((abs(x) / x_0)**2 + (abs(y) / y_0)**2)**M * \
                #             exp(-0.5 * ((abs(x) / x_0) ** 2 + (abs(y) / y_0) ** 2)) * \
                #             exp(1j * m * (arctan2(x, y) + pi))

        return arr

    def save_field(self, path, only_center=True):
        if only_center:
            percent = 3
            center = self.__n_x // 2
            ambit = int(self.__n_x * percent / 100)
            field = self._field[center-ambit:center+ambit, center-ambit:center+ambit]
        else:
            field = self._field
        save(path, field)
