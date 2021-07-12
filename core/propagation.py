from numpy import zeros
from numba import jit
from numpy import save

from .visualization import BeamVisualizer, plot_track, plot_noise
from .logger import Logger
from .manager import Manager


class Propagator:
    """
    Сlass describes the propagation of a laser beam in a medium. It accumulates a large number of objects of other
    classes and is one of the key ones in the program.
    """

    def __init__(self, **kwargs):
        self.__beam = kwargs['beam']  # beam object
        self.__diffraction = kwargs.get('diffraction', None)  # diffraction object
        self.__kerr_effect = kwargs.get('kerr_effect', None)  # kerr effect object

        self.__args = kwargs['args']  # command line arguments
        self.__multidir_name = kwargs.get('multidir_name', None)  # multidir name if used
        self.__save_field = kwargs.get('save_field', False)
        self.__manager = Manager(args=self.__args,                            #
                                 multidir_name=self.__multidir_name,          # manager object
                                 save_field=self.__save_field)  #
        self.__logger = Logger(diffraction=self.__diffraction,   #
                               kerr_effect=self.__kerr_effect,   # logger object
                               path=self.__manager.results_dir)  #

        self.__n_z = kwargs['n_z']  # maximum number of grid steps along evolutionary coordinate z
        self.__const_dz = kwargs['const_dz']  # use constant step along z or not

        self.__print_current_state_every = kwargs.get('print_current_state_every', None)  # frequency of current state print

        self.__plot_beam_every = kwargs.get('plot_beam_every', None)  # frequency of plotting beam
        self.__plot_spectrum_every = kwargs.get('plot_spectrum_every', None)  # frequency of plotting spectrum
        self.__flag_print_track = kwargs.get('print_track', True)  # print track function or not

        # settings for function which plots beam
        if self.__plot_beam_every:
            self.__visualizer = kwargs['visualizer']
            self.__visualizer.get_path_to_save(self.__manager.beam_dir)

        # spectrum
        self.__spectrum = kwargs.get('spectrum', None)
        if self.__plot_spectrum_every:
            self.__spectrum_visualizer = kwargs['spectrum_visualizer']
            self.__spectrum_visualizer.get_path_to_save(self.__manager.beam_dir)

        self.__z = 0.0  # initial value of z
        self.__dz = kwargs['dz_0']  # initial step along z

        self.__max_intensity_to_stop = kwargs.get('max_intensity_to_stop', 10**17)  # peak intensity in beam
                                                                                    # at which the calculations stop

        self.__states_columns = ['z, m', 'dz, m', 'i_max / i_0', 'i_max, W / m^2']  # columns for propagation file
        self.__states_arr = zeros(shape=(self.__n_z + 1, 4))  # array for states data

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
        """Flush current state data to states_arr"""

        states_arr[n_step][0] = z
        states_arr[n_step][1] = dz
        states_arr[n_step][2] = i_max / i_0
        states_arr[n_step][3] = i_max

    @staticmethod
    @jit(nopython=True)
    def __update_dz(k_0, n_0, n_2, i_max, dz, nonlin_phase_max=0.05):
        """
        Reduces the step along the evolutionary coordinate z by calculating the maximum Kerr phase incursion

        :return: updated dz
        """

        nonlin_phase = k_0 * n_2 * i_max * dz / n_0
        if nonlin_phase > nonlin_phase_max:
            dz *= 0.8 * nonlin_phase_max / nonlin_phase

        return dz

    def __crop_states_arr(self):
        """
        If the calculations end before reaching the value n_z, crops the remainder of the states_arr

        :return: cropped states_arr
        """

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
        """
        The main function of class Propagator. Realizes the propagation process of the beam.

        :return: None
        """
        # initial preparations
        self.__manager.create_dirs()
        self.__logger.save_initial_parameters(self.__beam, self.__n_z, self.__dz, self.__max_intensity_to_stop)
        if self.__beam.info == 'beam_xy' and self.__beam.noise_percent:
            plot_noise(self.__beam, self.__manager.results_dir)
            if self.__save_field:
                print(type(self.__beam.noise.noise_field))
                print(self.__beam.noise.noise_field.shape)
                save('{}/noise.npy'.format(self.__manager.results_dir), self.__beam.noise.noise_field)

        # main cycle
        for n_step in range(int(self.__n_z) + 1):
            if n_step:

                # diffraction
                if self.__diffraction:
                    self.__logger.measure_time(self.__diffraction.process_diffraction, [self.__dz])

                # kerr effect
                if self.__kerr_effect:
                    self.__logger.measure_time(self.__kerr_effect.process_kerr_effect, [self.__dz])

                # increase evolutionary coordinate z by current step
                self.__z += self.__dz

                # update intensity and step along z (if needed)
                self.__logger.measure_time(self.__beam.update_intensity, [])
                if not self.__const_dz:
                    self.__dz = self.__logger.measure_time(self.__update_dz, [self.__beam.medium.k_0,
                                                                              self.__beam.medium.n_0,
                                                                              self.__beam.medium.n_2,
                                                                              self.__beam.i_max,
                                                                              self.__dz])

            # flush current state
            self.__logger.measure_time(self.__flush_current_state, [self.__states_arr, n_step, self.__z, self.__dz,
                                                                    self.__beam.i_max, self.beam.i_0])

            # print current state
            if self.__print_current_state_every:
                if not n_step % self.__print_current_state_every:
                    self.__logger.measure_time(self.__logger.print_current_state, [n_step, self.__states_arr,
                                                                                   self.__states_columns])

            # plot beam
            if self.__plot_beam_every and not (n_step % self.__plot_beam_every):
                self.__logger.measure_time(self.__visualizer.plot_beam, [self.__beam, self.__z, n_step])

            # plot spectrum
            if self.__plot_spectrum_every and not (n_step % self.__plot_spectrum_every):
                self.__logger.measure_time(self.__spectrum.update, [self.__beam])
                self.__logger.measure_time(self.__spectrum_visualizer.plot, [self.__spectrum, self.__z, n_step])

            # save field
            if self.__save_field:
                path = self.__manager.field_dir + '/%04d' % n_step
                self.__beam.save_field(path)

            # check if calculations must be stopped
            if self.__beam.i_max > self.__max_intensity_to_stop:
                break

        # cropped states arr and log track
        self.__logger.measure_time(self.__crop_states_arr, [])
        self.__logger.measure_time(self.__logger.log_track, [self.__states_arr, self.__states_columns])

        # print track
        if self.__flag_print_track:
            parameter_index = self.__states_columns.index('i_max / i_0')
            self.__logger.measure_time(plot_track, [self.__states_arr, parameter_index,
                                                    self.__manager.track_dir])

        # log time of all functions
        self.__logger.log_times()
