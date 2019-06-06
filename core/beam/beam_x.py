from numpy import exp, zeros, complex64, heaviside

from .beam_2d import Beam2D


class BeamX(Beam2D):
    """
    Subsubclass for 2-dimensional beam with transverse coordinate x
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__x_0 = kwargs['x_0']  # characteristic spatial size
        self.__x_max = 40.0 * self.__x_0  # spatial grid size
        self.__n_x = kwargs['n_x']  # number of points in spatial grid
        self.__dx = self.__x_max / self.__n_x  # spatial grid step
        self.__xs = [i * self.__dx - 0.5 * self.__x_max for i in range(self.__n_x)]  # spatial grid nodes

        # field initialization
        self._field = self.__initialize_field(self._half, self._M, self.__x_0, self.__x_max, self.__dx, self.__n_x)

        # other parameters initialization
        self._z_diff = self.medium.k_0 * self.__x_0 ** 2
        self._r_kerr = kwargs.get('r_kerr', 100)

        # initial intensity in 2-dimensional beam is calculated from value of r_kerr!!!
        self._i_0 = 0.5 * self._r_kerr * self.medium.n_0 / (self.medium.k_0 * self.medium.n_2 * self._z_diff)

        self.update_intensity()

    @property
    def info(self):
        return 'beam_x'

    @property
    def x_0(self):
        return self.__x_0

    @property
    def x_max(self):
        return self.__x_max

    @property
    def n_x(self):
        return self.__n_x

    @property
    def xs(self):
        return self.__xs

    @property
    def dx(self):
        return self.__dx

    @staticmethod
    def __initialize_field(half, M, x_0, x_max, dx, n_x):
        """
        :param half: flag to use only half of the distribution
        :param M: power of polynomial before exponent in initial condition
        :param x_0: characteristic spatial size
        :param x_max: spatial grid size
        :param dx: spatial grid step
        :param n_x: number of points in spatial grid

        :return: initialized field array
        """
        arr = zeros(shape=(n_x,), dtype=complex64)
        for i in range(n_x):
            x = i * dx - 0.5 * x_max
            arr[i] = ((heaviside(-x, 0) * (1 - half) + heaviside(x, 0)) * (abs(x) / x_0)) ** M * \
                     exp(-0.5 * (abs(x) / x_0) ** 2)

        return arr
