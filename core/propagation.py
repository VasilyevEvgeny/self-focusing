from numpy import zeros, multiply
from numba import jit

from .visualization import plot_beam_2d, plot_beam_3d_flat, plot_beam_3d_volume, plot_track, plot_noise
from .logger import Logger
from .manager import Manager


class Propagator:
    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']
        self.__diffraction = kwargs.get('diffraction', None)
        self.__kerr_effect = kwargs.get('kerr_effect', None)

        self.__args = kwargs['args']
        self.__multidir_name = kwargs.get('multidir_name', None)
        self.__manager = Manager(args=self.__args,
                                 multidir_name=self.__multidir_name)
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
            self.__beam_normalization_type = kwargs['beam_normalization_type']
            self.__plot_beam_func = kwargs.get('plot_beam_func', None)
            if self.__plot_beam_func is None:
                if self.__beam.info == 'beam_x':
                    self.__plot_beam_func = plot_beam_2d
                else:
                    self.__beam_in_3D = kwargs.get('beam_in_3D', False)
                    if self.__beam_in_3D:
                        self.__plot_beam_func = plot_beam_3d_volume
                    else:
                        self.__plot_beam_func = plot_beam_3d_flat

        self.__z = 0.0
        self.__dz = kwargs['dz0']

        self.__max_intensity_to_stop = kwargs.get('max_intensity_to_stop', 10**17)

        self.__states_columns = ['z, m', 'dz, m', 'i_max / i_0', 'i_max, W / m^2']
        self.__states_arr = zeros(shape=(self.__n_z + 1, 4))

    @property
    def beam(self):
        return self.__beam

    @property
    def logger(self):
        return self.__logger

    @property
    def manager(self):
        return self.__manager

    @property
    def z(self):
        return self.__z

    @staticmethod
    @jit(nopython=True)
    def __flush_current_state(states_arr, n_step, z, dz, i_max, i_0):
        states_arr[n_step][0] = z
        states_arr[n_step][1] = dz
        states_arr[n_step][2] = i_max
        states_arr[n_step][3] = i_max * i_0

    @staticmethod
    @jit(nopython=True)
    def __update_dz(k_0, n_0, n_2, i_max, i_0, dz, nonlin_phase_max=0.05):
        nonlin_phase = k_0 * n_2 * i_0 * i_max * dz / n_0
        if nonlin_phase > nonlin_phase_max:
            dz *= 0.8 * nonlin_phase_max / nonlin_phase

        return dz

    def __crop_states_arr(self):
        row_max = 0
        for i in range(self.__states_arr.shape[0] - 1, 0, -1):
            if self.__states_arr[i][0] != 0 and \
                    self.__states_arr[i][1] != 0 and \
                    self.__states_arr[i][2] != 0 and \
                    self.__states_arr[i][3] != 0:
                row_max = i + 1
                break

        self.__states_arr = self.__states_arr[:row_max, :]

    def propagate(self):
        self.__manager.create_dirs()
        self.__logger.save_initial_parameters(self.__beam, self.__n_z, self.__dz, self.__max_intensity_to_stop)
        if self.__beam.info == 'beam_xy' and self.__beam.noise_percent:
            plot_noise(self.__beam, self.__manager.results_dir)

        # main cycle
        for n_step in range(int(self.__n_z) + 1):
            if n_step:
                if self.__diffraction:
                    self.__logger.measure_time(self.__diffraction.process_diffraction, [self.__dz])

                if self.__kerr_effect:
                    self.__logger.measure_time(self.__kerr_effect.process_kerr_effect, [self.__dz])

                self.__logger.measure_time(self.__beam.update_intensity, [])

                self.__z += self.__dz

                if not self.__flag_const_dz:
                    self.__dz = self.__logger.measure_time(self.__update_dz, [self.__beam.medium.k_0, self.__beam.medium.n_0,
                                                                              self.__beam.medium.n_2, self.__beam.i_max,
                                                                              self.__beam.i_0, self.__dz])

            self.__logger.measure_time(self.__flush_current_state, [self.__states_arr, n_step, self.__z, self.__dz,
                                                                    self.__beam.i_max, self.__beam.i_0])

            if self.__dn_print_current_state:
                if not n_step % self.__dn_print_current_state:
                    self.__logger.measure_time(self.__logger.print_current_state, [n_step, self.__states_arr,
                                                                                   self.__states_columns])

            if self.__dn_plot_beam and not (n_step % self.__dn_plot_beam):
                self.__logger.measure_time(self.__plot_beam_func, [self.__args.prefix, self.__beam, self.__z, n_step,
                                                                   self.__manager.beam_dir,
                                                                   self.__beam_normalization_type])

            if self.__beam.i_max * self.__beam.i_0 > self.__max_intensity_to_stop:
                break

        self.__logger.measure_time(self.__crop_states_arr, [])
        self.__logger.measure_time(self.__logger.log_track, [self.__states_arr, self.__states_columns])

        if self.__flag_print_track:
            parameter_index = self.__states_columns.index('i_max / i_0')
            self.__logger.measure_time(plot_track, [self.__states_arr, parameter_index,
                                                    self.__manager.track_dir])

        self.__logger.log_times()
