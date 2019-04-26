from core.libs import *


class FourierDiffractionExecutor_XY:
    max_number_of_cpus = cpu_count()

    def __init__(self, **kwargs):
        self.beam = kwargs["beam"]

        self.k_xs_arr = np.full(self.beam.field.shape, self.beam.k_xs)
        self.k_ys_arr = np.full(self.beam.field.shape, self.beam.k_ys).T

    @property
    def info(self):
        return "fourier_diffraction_executor_xy"

    # @staticmethod
    # @jit(nopython=True)
    # def phase_increment(field_fft, n_x, n_y, k_xs, k_ys, current_lin_phase):
    #     for i in range(n_x):
    #         field_fft[i, :] *= exp(current_lin_phase * k_xs[i] ** 2)
    #
    #     for j in range(n_y):
    #         field_fft[:, j] *= exp(current_lin_phase * k_ys[j] ** 2)
    #
    #     return field_fft
    #
    @staticmethod
    @jit(nopython=True)
    def phase_increment2(field_fft, n_x, n_y, k_xs, k_ys, current_lin_phase):
        for i in range(n_x):
            for j in range(n_y):
                field_fft[i, j] *= exp(current_lin_phase * (k_xs[i] ** 2 + k_ys[j] ** 2))

        return field_fft

    # @staticmethod
    # def phase_increment(field_fft, current_lin_phase, k_xs_arr, k_ys_arr):
    #     field_fft = np.multiply(field_fft, exp(current_lin_phase * (k_xs_arr ** 2 + k_ys_arr ** 2)))
    #
    #    return field_fft

    # def fft_r(arr, m=m, s_step=s_step):
    #     for s in range(0, arr.shape[1], s_step):
    #         arr[:, s] = np.fft.fft2(r_to_xy(arr[:, s], m))[:arr.shape[0], 0]
    #
    #     return arr

    # @staticmethod
    # def make_increment(arr):
    #     global current_lin_phase, k_xs, k_ys
    #     #for j in range(arr.shape[1]):
    #     #    arr[:, j] *= exp(current_lin_phase * (kx_s[:]**2 + ky_s[j]**2))
    #
    #     return arr

    # @staticmethod
    # def init(args):
    #     global current_lin_phase, k_xs, k_ys
    #     current_lin_phase = args[0]
    #     k_xs = args[1]
    #     k_ys = args[2]

    # def phase_increment(self, field_fft, current_lin_phase, k_xs, k_ys, n_jobs=max_number_of_cpus):
    #     # idx = 0
    #     # arrs = []
    #     length = int(field_fft.shape[1] / n_jobs)
    #     # for j in range(n_jobs):
    #     #     arrs.append(field_fft[:, idx:idx + length]) if j != n_jobs - 1 else arrs.append(field_fft[:, idx:])
    #     #     idx += length
    #
    #     pool = Pool(processes=n_jobs,
    #                 initializer=self.init((current_lin_phase, k_xs, k_ys)),
    #                 initargs=(current_lin_phase, k_xs, k_ys,))
    #     res = pool.map_async(self.make_increment, [field_fft[:, j * length:j*length+length] if j != n_jobs - 1 else field_fft[:, j*length:]
    #                                                for j in range(n_jobs)])
    #     res.wait()
    #     result = res.get()
    #     #for e in result:
    #     #    print(e.shape)
    #     field_fft = np.concatenate(result, axis=1)
    #
    #     print("done!")
    #
    #
    #     return field_fft

    def process_diffraction(self, dz, n_jobs=max_number_of_cpus):
        current_lin_phase = 0.5j * dz / self.beam.medium.k_0
        fft_obj = fft2(self.beam.field, threads=n_jobs)
        field_fft = self.phase_increment2(fft_obj(), self.beam.n_x, self.beam.n_y, self.beam.k_xs,
                                          self.beam.k_ys, current_lin_phase)
        #field_fft = self.phase_increment2(fft_obj(), current_lin_phase, self.k_xs_arr, self.k_ys_arr)
        ifft_obj = ifft2(field_fft, threads=n_jobs)
        self.beam.field = ifft_obj()


class SweepDiffractionExecutor_R:
    def __init__(self, **kwargs):
        self.beam = kwargs["beam"]
        self.c1 = 1.0 / (2.0 * self.beam.dr ** 2)
        self.c2 = 1.0 / (4.0 * self.beam.dr)
        self.c3 = 2j * self.beam.medium.k_0

        self.alpha = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)
        self.beta = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)
        self.gamma = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)
        self.vx = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)

        for i in range(1, self.beam.n_r - 1):
            self.alpha[i] = self.c1 + self.c2 / self.beam.rs[i]
            self.gamma[i] = self.c1 - self.c2 / self.beam.rs[i]
            self.vx[i] = (self.beam.m / self.beam.rs[i]) ** 2

        self.kappa_left, self.mu_left, self.kappa_right, self.mu_right = \
            1.0, 0.0, 0.0, 0.0

        self.delta = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)
        self.xi = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)
        self.eta = np.zeros(shape=(self.beam.n_r,), dtype=np.complex64)

    @property
    def info(self):
        return "sweep_diffraction_executor_r"

    @staticmethod
    @jit(nopython=True)
    def fast_process(field, n_r, dz, c1, c3, alpha, beta, gamma, delta, xi, eta, vx,
                     kappa_left, mu_left, kappa_right, mu_right):
        xi[1], eta[1] = kappa_left, mu_left
        for i in range(1, n_r - 1):
            beta[i] = 2.0 * c1 + c3 / dz + vx[i]
            delta[i] = alpha[i] * field[i + 1] - \
                       (conj(beta[i]) - vx[i]) * field[i] + \
                       gamma[i] * field[i - 1]
            xi[i + 1] = alpha[i] / (beta[i] - gamma[i] * xi[i])
            eta[i + 1] = (delta[i] + gamma[i] * eta[i]) / \
                         (beta[i] - gamma[i] * xi[i])

        field[n_r - 1] = (mu_right + kappa_right * eta[n_r - 1]) / \
                         (1.0 - kappa_right * xi[n_r - 1])

        for j in range(n_r - 1, 0, -1):
            field[j - 1] = xi[j] * field[j] + eta[j]

        return field

    def process_diffraction(self, dz):
        self.beam.field = self.fast_process(self.beam.field, self.beam.n_r, dz, self.c1,
                                            self.c3, self.alpha, self.beta, self.gamma, self.delta, self.xi, self.eta,
                                            self.vx, self.kappa_left, self.mu_left, self.kappa_right, self.mu_right)
