from numpy import zeros, multiply
from numba import jit

from .visualization import plot_beam_2d, plot_beam_3d, plot_track, plot_noise
from .logger import Logger
from .manager import Manager


class Propagator:
    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__diffraction = kwargs.get('diffraction', None)
        self.__kerr_effect = kwargs.get('kerr_effect', None)

        self.__args = kwargs['args']
        self.__multidir_name = kwargs.get('multidir_name', None)
        self.__manager = Manager(args=self.__args, multidir_name=self.__multidir_name)
        self.__logger = Logger(diffraction=self.__diffraction,
                             kerr_effect=self.__kerr_effect,
                             path=self.__manager.results_dir)

        self.__n_z = kwargs['n_z']
        self.__flag_const_dz = kwargs['flag_const_dz']

        self.__dn_print_current_state = kwargs.get('dn_print_current_state', None)
        self.__flag_print_beam = True if self.__dn_print_current_state else False

        self.__dn_plot_beam = kwargs.get('dn_plot_beam', None)
        self.__flag_print_track = kwargs.get('print_track', True)
        if self.__dn_plot_beam:
            self.beam_normalization_type = kwargs['beam_normalization_type']

        self.z = 0.0
        self.dz = kwargs['dz0']

        self.max_intensity_to_stop = 10**18

        self.states_columns = ['z, m', 'dz, m', 'i_max / i_0', 'i_max, W / m^2']
        self.states_arr = zeros(shape=(self.__n_z + 1, 4))

    @property
    def beam(self):
        return self.__beam

    @property
    def logger(self):
        return self.__logger

    @property
    def manager(self):
        return self.__manager

    @staticmethod
    @jit(nopython=True)
    def flush_current_state(states_arr, n_step, z, dz, i_max, i_0):
        states_arr[n_step][0] = z
        states_arr[n_step][1] = dz
        states_arr[n_step][2] = i_max
        states_arr[n_step][3] = i_max * i_0

    @staticmethod
    @jit(nopython=True)
    def update_dz(k_0, n_0, n_2, i_max, i_0, dz, nonlin_phase_max=0.05):
        nonlin_phase = k_0 * n_2 * i_0 * i_max * dz / n_0
        if nonlin_phase > nonlin_phase_max:
            dz *= 0.8 * nonlin_phase_max / nonlin_phase

        return dz

    def crop_states_arr(self):
        row_max = 0
        for i in range(self.states_arr.shape[0] - 1, 0, -1):
            if self.states_arr[i][0] != 0 and \
                    self.states_arr[i][1] != 0 and \
                    self.states_arr[i][2] != 0 and \
                    self.states_arr[i][3] != 0:
                row_max = i + 1
                break

        self.states_arr = self.states_arr[:row_max, :]

    def apply_phase_noise_screen_to_field(self):
        self.__beam.field = multiply(self.__beam.field, self.__beam.phase_noise_screen)

    def propagate(self):
        self.__manager.create_global_results_dir()
        self.__manager.create_results_dir()
        self.__manager.create_track_dir()
        self.__manager.create_beam_dir()

        self.__logger.save_initial_parameters(self.__beam, self.__n_z, self.dz, self.max_intensity_to_stop)

        if self.__beam.info == 'beam_xy' and self.__beam.noise_percent:
            plot_noise(self.__beam, self.__manager.results_dir)

        for n_step in range(int(self.__n_z) + 1):
            if n_step:
                if self.__diffraction:
                    self.__logger.measure_time(self.__diffraction.process_diffraction, [self.dz])

                if self.__kerr_effect:
                    self.__logger.measure_time(self.__kerr_effect.process_kerr_effect, [self.dz])

                self.__logger.measure_time(self.__beam.update_intensity, [])

                self.z += self.dz

                if not self.__flag_const_dz:
                    self.dz = self.__logger.measure_time(self.update_dz, [self.__beam.medium.k_0, self.__beam.medium.n_0,
                                                                          self.__beam.medium.n_2, self.__beam.i_max,
                                                                          self.__beam.i_0, self.dz])

            self.__logger.measure_time(self.flush_current_state, [self.states_arr, n_step, self.z, self.dz,
                                                                  self.__beam.i_max, self.__beam.i_0])

            if self.__dn_print_current_state:
                if not n_step % self.__dn_print_current_state:
                    self.__logger.measure_time(self.__logger.print_current_state, [n_step, self.states_arr,
                                                                                   self.states_columns])

            if self.__dn_plot_beam:
                if (not (n_step % self.__dn_plot_beam)) and self.__flag_print_beam:
                    if self.__beam.info == 'beam_x':
                        self.__logger.measure_time(plot_beam_2d, [self.__args.prefix, self.__beam, self.z, n_step,
                                                                  self.__manager.beam_dir, self.beam_normalization_type])
                    elif self.__beam.info in ('beam_r', 'beam_xy'):
                        self.__logger.measure_time(plot_beam_3d, [self.__args.prefix, self.__beam, self.z, n_step,
                                                                  self.__manager.beam_dir, self.beam_normalization_type])

            if self.__beam.i_max * self.__beam.i_0 > self.max_intensity_to_stop:
                break

        self.__logger.measure_time(self.crop_states_arr, [])
        self.__logger.measure_time(self.__logger.log_track, [self.states_arr, self.states_columns])

        if self.__flag_print_track:
            parameter_index = self.states_columns.index('i_max / i_0')
            self.__logger.measure_time(plot_track, [self.states_arr, parameter_index,
                                                    self.__manager.track_dir])

        self.__logger.log_times()
