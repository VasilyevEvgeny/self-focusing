from core.libs import *
from core.medium import Medium
from core.m_constants import M_Constants


class Beam(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.m_constants = M_Constants()
        self.lmbda = kwargs["lmbda"]
        self.omega = 2.0 * pi * self.m_constants.c / self.lmbda

        self.medium = Medium(name=kwargs["medium"],
                             lmbda=self.lmbda,
                             m_constants=self.m_constants)

        self.Pcr_G = self.calculate_Pcr_G()

        self.M, self.m = None, None
        self.field = None
        self.distribution_type = kwargs["distribution_type"]
        if self.distribution_type == "gauss":
            self.m, self.M = 0, 0
            self.P0_to_Pcr_G = kwargs["P0_to_Pcr_G"]
            self.p_0 = self.P0_to_Pcr_G * self.Pcr_G
        elif self.distribution_type == "ring":
            self.m = 0
            self.M = kwargs["M"]
            self.P0_to_Pcr_G = kwargs["P0_to_Pcr_G"]
            self.p_0 = self.P0_to_Pcr_G * self.Pcr_G
        elif self.distribution_type == "vortex":
            self.m = kwargs["m"]
            self.M = self.m
            self.amp_noise_coeff = kwargs["amp_noise_coeff"]
            self.phase_noise_coeff = kwargs["phase_noise_coeff"]
            self.Pcr_V = self.calculate_Pcr_V()
            self.P0_to_Pcr_V = kwargs["P0_to_Pcr_V"]
            self.p_0 = self.P0_to_Pcr_V * self.Pcr_V
        else:
            raise Exception("Wrong distribution type: '%s'." % self.distribution_type)

        self.intensity, self.i_max = None, None

    @abc.abstractmethod
    def info(self):
        """Information about beam"""

    def calculate_Pcr_G(self):
        return 3.77 * self.lmbda ** 2 / (8 * pi * self.medium.n_0 * self.medium.n_2)

    def calculate_Pcr_V(self):
        return self.Pcr_G * 2**(2 * self.m + 1) * gamma(self.m + 1) * gamma(self.m + 2) / (2 * gamma(2 * self.m + 1))


class Beam_R(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.r_0 = kwargs["r_0"]
        self.r_max = 40.0 * self.r_0
        self.n_r = kwargs["n_r"]
        self.dr = self.r_max / self.n_r
        self.rs = [i * self.dr for i in range(self.n_r)]
        self.dk_r = 2.0 * pi / self.r_max
        self.k_rs = [i * self.dk_r for i in range(self.n_r)]

        if self.distribution_type == "gauss":
            self.field = self.initialize_field_gauss(self.r_0, self.dr, self.n_r)
        elif self.distribution_type == "ring":
            self.field = self.initialize_field_ring(self.M, self.r_0, self.dr, self.n_r)
        elif self.distribution_type == "vortex":
            self.field = self.initialize_field_vortex(self.m, self.r_0, self.dr, self.n_r)

        self.i_0 = self.calculate_I0()
        self.z_diff = self.medium.k_0 * self.r_0**2

        self.update_intensity()

    @property
    def info(self):
        return "beam_r"

    def update_intensity(self):
        self.intensity = self.field_to_intensity(self.field, self.n_r)
        self.i_max = np.max(self.intensity)

    @staticmethod
    @jit(nopython=True)
    def field_to_intensity(field, n_r):
        intensity = np.zeros(shape=(n_r,), dtype=np.float64)
        for i in range(n_r):
            intensity[i] = (field[i] * field[i].conjugate()).real

        return intensity

    def calculate_I0(self):
        return self.p_0 / (pi * self.r_0**2 * gamma(self.m+1))

    @staticmethod
    @jit(nopython=True, debug=True)
    def initialize_field_gauss(r_0, dr, n_r):
        arr = np.zeros(shape=(n_r,), dtype=np.complex64)
        for i in range(n_r):
            r = i * dr
            arr[i] = exp(-0.5 * (r ** 2 / r_0 ** 2))

        return arr

    @staticmethod
    @jit(nopython=True)
    def initialize_field_ring(M, r_0, dr, n_r):
        arr = np.zeros(shape=(n_r,), dtype=np.complex64)
        for i in range(n_r):
            r = i * dr
            arr[i] = (r / r_0)**M * exp(-0.5 * (r**2 / r_0**2))

        return arr

    def initialize_field_vortex(self, m, r_0, dr, n_r):
        return self.initialize_field_ring(m, r_0, dr, n_r)


class Beam_XY(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x_0 = kwargs["x_0"]
        self.y_0 = kwargs["y_0"]

        self.x_max = 20.0 * max(self.x_0, self.y_0)
        self.y_max = self.x_max

        self.n_x = kwargs["n_x"]
        self.n_y = kwargs["n_y"]

        self.dx = self.x_max / self.n_x
        self.dy = self.y_max / self.n_y

        self.xs = [i * self.dx - 0.5 * self.x_max for i in range(self.n_x)]
        self.ys = [i * self.dy - 0.5 * self.y_max for i in range(self.n_y)]

        self.dk_x = 2.0 * pi / self.x_max
        self.dk_y = 2.0 * pi / self.y_max

        self.k_xs = [i * self.dk_x if i < self.n_x / 2 else (i - self.n_x) * self.dk_x for i in range(self.n_x)]
        self.k_ys = [i * self.dk_y if i < self.n_y / 2 else (i - self.n_y) * self.dk_y for i in range(self.n_y)]

        if self.distribution_type == "gauss":
            self.field = self.initialize_field_gauss(self.x_0, self.y_0, self.x_max, self.y_max, self.dx, self.dy,
                                                     self.n_x, self.n_y)
        elif self.distribution_type == "ring":
            self.field = self.initialize_field_ring(self.M, self.x_0, self.y_0, self.x_max, self.y_max, self.dx,
                                                    self.dy, self.n_x, self.n_y)
        elif self.distribution_type == "vortex":
            self.field = self.initialize_field_vortex(self.m, self.x_0, self.y_0, self.x_max, self.y_max, self.dx,
                                                      self.dy, self.n_x, self.n_y, self.amp_noise_coeff,
                                                      self.phase_noise_coeff)

        self.i_0 = self.calculate_I0()
        self.z_diff = self.medium.k_0 * (self.x_0 ** 2 + self.y_0 ** 2) / 2

        self.update_intensity()

    @property
    def info(self):
        return "beam_xy"

    def update_intensity(self):
        self.intensity = self.field_to_intensity(self.field, self.n_x, self.n_y)
        self.i_max = np.max(self.intensity)

    @staticmethod
    @jit(nopython=True)
    def field_to_intensity(field, n_x, n_y):
        intensity = np.zeros(shape=(n_x, n_y), dtype=np.float64)
        for i in range(n_x):
            for j in range(n_y):
                intensity[i, j] = (field[i, j] * field[i, j].conjugate()).real

        return intensity

    def calculate_I0(self):
        # i_distribution = lambda x, y: mpmath.norm(self.a_distribution(x, y))**2
        # lim = mpf(max(self.x_max, self.y_max))
        # i_0 = self.p_0 / float(mpmath_nquad(lambda x, y: i_distribution(x, y), [-lim, +lim], [-lim, +lim]))

        return self.p_0 / (pi * (self.x_0**2 + self.y_0**2) / 2 * gamma(self.m+1))

    @staticmethod
    @jit(nopython=True, debug=True)
    def initialize_field_gauss(x_0, y_0, x_max, y_max, dx, dy, n_x, n_y):
        arr = np.zeros(shape=(n_x, n_y), dtype=np.complex64)
        for i in range(n_x):
            for j in range(n_y):
                x, y = i * dx - 0.5 * x_max, j * dy - 0.5 * y_max
                arr[i, j] = exp(-0.5 * (x ** 2 / x_0 ** 2 + y ** 2 / y_0 ** 2))

        return arr

    @staticmethod
    @jit(nopython=True)
    def initialize_field_ring(M, x_0, y_0, x_max, y_max, dx, dy, n_x, n_y):
        arr = np.zeros(shape=(n_x, n_y), dtype=np.complex64)
        for i in range(n_x):
            for j in range(n_y):
                x, y = i * dx - 0.5 * x_max, j * dy - 0.5 * y_max
                arr[i, j] = sqrt(x ** 2 / x_0 ** 2 + y ** 2 / y_0 ** 2) ** M * exp(
                    -0.5 * (x ** 2 / x_0 ** 2 + y ** 2 / y_0 ** 2))

        return arr

    @staticmethod
    @jit(nopython=False)
    def initialize_field_vortex(m, x_0, y_0, x_max, y_max, dx, dy, n_x, n_y, amp_noise_coeff, phase_noise_coeff):
        arr = np.zeros(shape=(n_x, n_y), dtype=np.complex64)
        for i in range(n_x):
            for j in range(n_y):
                x, y = i * dx - 0.5 * x_max, j * dy - 0.5 * y_max
                da = 2 * (np.random.random() - 0.5) * amp_noise_coeff
                dphi = 2 * (np.random.random() - 0.5) * phase_noise_coeff
                arr[i, j] = (1 + da) * sqrt(x**2 / x_0**2 + y**2 / y_0**2)**m * exp(-0.5 * (x ** 2 / x_0 ** 2 + y ** 2 / y_0 ** 2)) * \
                            exp(1j * m * arctan2(x, y) * (1.0 + dphi))

        return arr
