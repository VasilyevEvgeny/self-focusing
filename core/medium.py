from numpy import sqrt, pi


class Medium:
    def __init__(self, **kwargs):
        self.__m_constants = kwargs['m_constants']
        self.__c = self.__m_constants.c
        self.__lmbda = kwargs['lmbda']
        self.__name = kwargs['name']

        self.__n_0, self.__k_0, self.__k_1, self.__k_2, self.__n_2 = None, None, None, None, None
        if self.__name == 'SiO2':
            self.initialize_SiO2()
            self.__n_2 = 3.4 * 10**-20
        elif self.__name == 'CaF2':
            self.initialize_CaF2()
            self.__n_2 = 1.92 * 10**-20
        elif self.__name == 'LiF':
            self.initialize_LiF()
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
    def calculate_n(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3):
        return sqrt(1 +
                    C_1 / (1 - (omega / omega_1) ** 2) +
                    C_2 / (1 - (omega / omega_2) ** 2) +
                    C_3 / (1 - (omega / omega_3) ** 2))

    def calculate_k_0(self, omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, c):
        return omega / c * self.calculate_n(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3)

    @staticmethod
    def calculate_k_1(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, c):
        return omega * (C_1 * omega / (omega_1 ** 2 * (-omega ** 2 / omega_1 ** 2 + 1) ** 2) +
               C_2 * omega / (omega_2 ** 2 * (-omega ** 2 / omega_2 ** 2 + 1) ** 2) +
               C_3 * omega / (omega_3 ** 2 * (-omega ** 2 / omega_3 ** 2 + 1) ** 2)) / \
               (c * sqrt(C_1 / (-omega ** 2 / omega_1 ** 2 + 1) + C_2 / (-omega ** 2 / omega_2 ** 2 + 1) +
               C_3 / (-omega ** 2 / omega_3 ** 2 + 1) + 1)) + sqrt(C_1 / (-omega ** 2 / omega_1 ** 2 + 1) +
               C_2 / (-omega ** 2 / omega_2 ** 2 + 1) + C_3 / (-omega ** 2 / omega_3 ** 2 + 1) + 1) / c

    @staticmethod
    def calculate_k_2(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, c):
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

    def initialize_parameters(self, lambda_1, lambda_2, lambda_3, C_1, C_2, C_3):
        omega_1, omega_2, omega_3 = 2. * pi * self.__c / lambda_1, 2. * pi * self.__c / lambda_2, 2. * pi * self.__c / lambda_3
        omega = 2 * pi * self.__c / self.__lmbda

        self.__n_0 = self.calculate_n(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3)
        self.__k_0 = self.calculate_k_0(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, self.__c)
        self.__k_1 = self.calculate_k_1(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, self.__c)
        self.__k_2 = self.calculate_k_2(omega, C_1, C_2, C_3, omega_1, omega_2, omega_3, self.__c)

    def initialize_SiO2(self):
        C_1 = 0.6961663000
        C_2 = 0.4079426000
        C_3 = 0.8974794000
        lambda_1 = 0.0684043000 * 10 ** -6
        lambda_2 = 0.1162414000 * 10 ** -6
        lambda_3 = 9.8961610000 * 10 ** -6

        self.initialize_parameters(lambda_1, lambda_2, lambda_3, C_1, C_2, C_3)

    def initialize_CaF2(self):
        C_1 = 0.5675888000
        C_2 = 0.4710914000
        C_3 = 3.8484723000
        lambda_1 = 0.0502636050 * 10 ** -6
        lambda_2 = 0.1003909000 * 10 ** -6
        lambda_3 = 34.649040000 * 10 ** -6

        self.initialize_parameters(lambda_1, lambda_2, lambda_3, C_1, C_2, C_3)

    def initialize_LiF(self):
        C_1 = 0.92549
        C_2 = 6.96747
        C_3 = 0
        lambda_1 = 0.0737600000 * 10 ** -6
        lambda_2 = 32.790000000 * 10 ** -6
        lambda_3 = 10**-10

        self.initialize_parameters(lambda_1, lambda_2, lambda_3, C_1, C_2, C_3)
