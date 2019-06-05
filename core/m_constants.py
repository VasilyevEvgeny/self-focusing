class MathConstants:
    def __init__(self):
        self.__h_bar = 1.054571800*10**-34
        self.__c = 2.9979245799999954*10**8
        self.__e = 1.602176620898*10**-19

    @property
    def h_bar(self):
        return self.__h_bar

    @property
    def c(self):
        return self.__c

    @property
    def e(self):
        return self.__e
