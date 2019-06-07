from numpy import sqrt, pi


class Medium:
    """
    Сlass that describes the object of the medium in which the beam propagates
    """
    def __init__(self, **kwargs):
        self.__m_constants = kwargs['m_constants']  # mathematical constants
        self.__c = self.__m_constants.c  # light speed in vacuum
        self.__lmbda = kwargs['lmbda']  # wavelength
        self.__name = kwargs['name']  # name of medium

        self.__n_0 = None  # linear refractive index
        self.__k_0 = None  # wave vector
        self.__k_1 = None  # dk/dw, s/m
        self.__k_2 = None  # d^2 k / dw^2, s^2/m
        self.__n_2 = None  # nonlinear refractive index by intensity, m^2/W

        # initalize medium parameters
        # it is assumed that n_2 is independent of wavelength!
        if self.__name == 'SiO2':
            self.__initialize_dispersion_parameters_SiO2()
            self.__n_2 = 3.4 * 10**-20
        elif self.__name == 'CaF2':
            self.__initialize_dispersion_parameters_CaF2()
            self.__n_2 = 1.92 * 10**-20
        elif self.__name == 'LiF':
            self.__initialize_dispersion_parameters_LiF()
            self.__n_2 = 1.0 * 10 ** -20
        else:
            raise Exception('Wrong name!')

    @property
    def info(self):
        return self.__name

    @property
    def n_0(self):
        return self.__n_0

    @property
    def k_0(self):
        return self.__k_0

    @property
    def k_1(self):
        return self.__k_1

    @property
    def k_2(self):
        return self.__k_2

    @property
    def n_2(self):
        return self.__n_2

    @staticmethod
    def __calculate_n(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3):
        """
        Linear refractive index is calculated from the Sellmeier formula
        (//: https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson)

        :param omega: beam frequency, \omega = 2 \pi c / \lambda
        :param C_1: parameter in Sellmeier formula
        :param C_2: parameter in Sellmeier formula
        :param C_3: parameter in Sellmeier formula
        :param omega_1: resonance frequency
        :param omega_2: resonance frequency
        :param omega_3: resonance frequency
        :return:
        """
        return sqrt(1 +
                    C_1 / (1 - (omega / omega_1) ** 2) +
                    C_2 / (1 - (omega / omega_2) ** 2) +
                    C_3 / (1 - (omega / omega_3) ** 2))

    def __calculate_k_0(self, omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, c):
        """
        Calculates wave vector
        k_0 = \omega * n(\omega) / c
        """

        return omega / c * self.__calculate_n(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3)

    @staticmethod
    def __calculate_k_1(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, c):
        """
        Calculates k_1 = dk / dw |_w=\omega using the analytical expression of the first derivative
        of the Sellmeier formula
        """

        return omega * (C_1 * omega / (omega_1 ** 2 * (-omega ** 2 / omega_1 ** 2 + 1) ** 2) +
               C_2 * omega / (omega_2 ** 2 * (-omega ** 2 / omega_2 ** 2 + 1) ** 2) +
               C_3 * omega / (omega_3 ** 2 * (-omega ** 2 / omega_3 ** 2 + 1) ** 2)) / \
               (c * sqrt(C_1 / (-omega ** 2 / omega_1 ** 2 + 1) + C_2 / (-omega ** 2 / omega_2 ** 2 + 1) +
               C_3 / (-omega ** 2 / omega_3 ** 2 + 1) + 1)) + sqrt(C_1 / (-omega ** 2 / omega_1 ** 2 + 1) +
               C_2 / (-omega ** 2 / omega_2 ** 2 + 1) + C_3 / (-omega ** 2 / omega_3 ** 2 + 1) + 1) / c

    @staticmethod
    def __calculate_k_2(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, c):
        """
        Calculates k_2 = d^2k / dw^2 |_w=\omega using the analytical expression of the second derivative
        of the Sellmeier formula
        """

        return omega * (-C_1 * omega / (omega_1 ** 2 * (-omega ** 2 / omega_1 ** 2 + 1) ** 2) - C_2 * omega / (
               omega_2 ** 2 * (-omega ** 2 / omega_2 ** 2 + 1) ** 2) - C_3 * omega / (omega_3 ** 2 * (-omega ** 2 /
               omega_3 ** 2 + 1) ** 2)) * (C_1 * omega / (omega_1 ** 2 * (-omega ** 2 / omega_1 ** 2 + 1) ** 2) +
               C_2 * omega / (omega_2 ** 2 * (-omega ** 2 / omega_2 ** 2 + 1) ** 2) + C_3 * omega / (omega_3 ** 2 *
               (-omega ** 2 / omega_3 ** 2 + 1) ** 2)) / (c * (C_1 / (-omega ** 2 / omega_1 ** 2 + 1) + C_2 /
               (-omega ** 2 / omega_2 ** 2 + 1) + C_3 / (-omega ** 2 / omega_3 ** 2 + 1) + 1) ** (3 / 2)) + \
               omega * (4 * C_1 * omega ** 2 / (omega_1 ** 4 * (-omega ** 2 / omega_1 ** 2 + 1) ** 3) + C_1 /
               (omega_1 ** 2 * (-omega ** 2 / omega_1 ** 2 + 1) ** 2) + 4 * C_2 * omega ** 2 / (omega_2 ** 4 *
               (-omega ** 2 / omega_2 ** 2 + 1) ** 3) + C_2 / (omega_2 ** 2 * (-omega ** 2 / omega_2 ** 2 + 1) ** 2) +
               4 * C_3 * omega ** 2 / (omega_3 ** 4 * (-omega ** 2 / omega_3 ** 2 + 1) ** 3) + C_3 / (omega_3 ** 2 *
               (-omega ** 2 / omega_3 ** 2 + 1) ** 2)) / (c * sqrt(C_1 / (-omega ** 2 / omega_1 ** 2 + 1) + C_2 /
               (-omega ** 2 / omega_2 ** 2 + 1) + C_3 / (-omega ** 2 / omega_3 ** 2 + 1) + 1)) + 2 * (C_1 * omega /
               (omega_1 ** 2 * (-omega ** 2 / omega_1 ** 2 + 1) ** 2) + C_2 * omega / (omega_2 ** 2 * (-omega ** 2 /
               omega_2 ** 2 + 1) ** 2) + C_3 * omega / (omega_3 ** 2 * (-omega ** 2 / omega_3 ** 2 + 1) ** 2)) / (c *
               sqrt(C_1 / (-omega ** 2 / omega_1 ** 2 + 1) + C_2 / (-omega ** 2 / omega_2 ** 2 + 1) + C_3 /
               (-omega ** 2 / omega_3 ** 2 + 1) + 1))

    def __initialize_dispersion_parameters(self, lambda_1, lambda_2, lambda_3, C_1, C_2, C_3):
        """
        Calculates all dispersion parameters in class
        """

        omega_1, omega_2, omega_3 = 2. * pi * self.__c / lambda_1, \
                                    2. * pi * self.__c / lambda_2, \
                                    2. * pi * self.__c / lambda_3
        omega = 2 * pi * self.__c / self.__lmbda

        self.__n_0 = self.__calculate_n(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3)
        self.__k_0 = self.__calculate_k_0(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, self.__c)
        self.__k_1 = self.__calculate_k_1(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, self.__c)
        self.__k_2 = self.__calculate_k_2(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, self.__c)

    def __initialize_dispersion_parameters_SiO2(self):
        """
        Initializes dispersion parameters for SiO2
        """

        C_1 = 0.6961663000
        C_2 = 0.4079426000
        C_3 = 0.8974794000
        lambda_1 = 0.0684043000 * 10 ** -6
        lambda_2 = 0.1162414000 * 10 ** -6
        lambda_3 = 9.8961610000 * 10 ** -6

        self.__initialize_dispersion_parameters(lambda_1, lambda_2, lambda_3, C_1, C_2, C_3)

    def __initialize_dispersion_parameters_CaF2(self):
        """
        Initializes dispersion parameters for CaF2
        """

        C_1 = 0.5675888000
        C_2 = 0.4710914000
        C_3 = 3.8484723000
        lambda_1 = 0.0502636050 * 10 ** -6
        lambda_2 = 0.1003909000 * 10 ** -6
        lambda_3 = 34.649040000 * 10 ** -6

        self.__initialize_dispersion_parameters(lambda_1, lambda_2, lambda_3, C_1, C_2, C_3)

    def __initialize_dispersion_parameters_LiF(self):
        """
        Initializes dispersion parameters for LiF
        """

        C_1 = 0.92549
        C_2 = 6.96747
        C_3 = 0
        lambda_1 = 0.0737600000 * 10 ** -6
        lambda_2 = 32.790000000 * 10 ** -6
        lambda_3 = 10**-10

        self.__initialize_dispersion_parameters(lambda_1, lambda_2, lambda_3, C_1, C_2, C_3)
