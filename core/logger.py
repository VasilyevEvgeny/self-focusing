from core.libs import *


class Logger:
    def __init__(self, **kwargs):
        self.path = kwargs["path"]
        self.diffraction = kwargs["diffraction"]
        self.kerr_effect = kwargs["kerr_effect"]

        self.functions = OrderedDict()

    def measure_time(self, function, args):
        t_start = time()
        res = function(*args)
        t_end = time()
        duration = t_end - t_start
        function_name = function.__name__
        if function_name in self.functions.keys():
            self.functions[function_name] += duration
        else:
            self.functions.update({function_name: 0.0})

        return res

    def log_times(self):
        with open(self.path + "/time_logs.txt", "w") as f:
            for key in self.functions:
                f.write("{:40s} = {:10}\n".format(key, str(timedelta(seconds=self.functions[key]))))

    def save_initial_parameters(self, beam):
        separator = "========================================"
        with open(self.path + "/parameters.txt", "w") as f:
            f.write(self.path + "\n")
            f.write(separator + "\n")
            f.write("EQUATION:\n")
            if self.diffraction is not None:
                f.write(self.diffraction.info + "\n")
            if self.kerr_effect is not None:
                f.write(self.kerr_effect.info + "\n")
            f.write(separator + "\n")
            f.write("MEDIUM:\n")
            f.write("{:20s} = {:s}\n".format("material", beam.medium.info))
            f.write("{:20s} = {:2.4f}\n".format("n_0", beam.medium.n_0))
            f.write("{:20s} = {:g}\n".format("k_0, m^-1", beam.medium.k_0))
            f.write("{:20s} = {:4.2f}\n".format("k_1, fs/mm", beam.medium.k_1 * 10**12))
            f.write("{:20s} = {:2.2f}\n".format("k_2, fs^2/mm", beam.medium.k_2 * 10**27))
            f.write("{:20s} = {:2.2f}\n".format("n_2, *10^-20 m^2/W", beam.medium.n_2 * 10**20))
            f.write(separator + "\n")
            f.write("BEAM:\n")
            f.write("{:20s} = {:s}\n".format("distribution", beam.distribution_type))
            if beam.distribution_type == "vortex":
                f.write("{:20s} = {:d}\n".format("m", beam.m))
            if beam.info == "beam_r":
                f.write("{:20s} = {:d}\n".format("r_0, microns", round(beam.r_0 * 10 ** 6)))
                f.write("{:20s} = {:d}\n".format("r_max, microns", round(beam.r_max * 10 ** 6)))
                f.write("{:20s} = {:d}\n".format("n_r", beam.n_r))
                f.write("{:20s} = {:2.2f}\n".format("dr, microns", beam.dr * 10 ** 6))
            if beam.info == "beam_xy":
                f.write("{:20s} = {:d}\n".format("x_0, microns", round(beam.x_0 * 10**6)))
                f.write("{:20s} = {:d}\n".format("y_0, microns", round(beam.y_0 * 10**6)))
                f.write("{:20s} = {:d}\n".format("x_max, microns", round(beam.x_max * 10**6)))
                f.write("{:20s} = {:d}\n".format("y_max, microns", round(beam.y_max * 10**6)))
                f.write("{:20s} = {:d}\n".format("n_x", beam.n_x))
                f.write("{:20s} = {:d}\n".format("n_y", beam.n_y))
                f.write("{:20s} = {:2.2f}\n".format("dx, microns", beam.dx * 10**6))
                f.write("{:20s} = {:2.2f}\n".format("dy, microns", beam.dy * 10**6))
                f.write("{:20s} = {:03.2f}\n".format("noise_percent", beam.noise_percent))
            f.write("{:20s} = {:d}\n".format("lmbda, nm", round(beam.lmbda * 10**9)))
            f.write("{:20s} = {:2.4f}\n".format("z_diff, m", beam.z_diff))
            if beam.distribution_type == "gauss" or beam.distribution_type == "ring":
                f.write("{:20s} = {:.2f}\n".format("p_0 / Pcr_G", beam.P0_to_Pcr_G))
            if beam.distribution_type == "vortex":
                f.write("{:20s} = {:.2f}\n".format("p_0 / Pcr_V", beam.P0_to_Pcr_V))
            f.write("{:20s} = {:.2f}\n".format("p_0, MW", beam.p_0 * 10 ** -6))
            f.write("{:20s} = {:e}\n".format("i_0, W/m^2", beam.i_0))

    @staticmethod
    def print_current_state(n_step, states, states_columns):
        if n_step == 0:
            print("      |   %s   |    %s   |  %s |  %s |" % (states_columns[0],
                                                         states_columns[1],
                                                         states_columns[2],
                                                         states_columns[3]))
        output_string = "{:04d} {:11.6f} {:13e} {:11.6f} {:17e}".format(n_step,
                                                                      states[n_step, 0],
                                                                      states[n_step, 1],
                                                                      states[n_step, 2],
                                                                      states[n_step, 3])
        print(output_string)

    def log_track(self, states_arr, states_columns, filename="propagation.xlsx"):
        filename = self.path + "/" + filename

        workbook = Workbook(filename)

        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True, 'align': 'center'})
        format_precise_general = workbook.add_format({'num_format': '###0.0000000', 'align': 'center'})
        format_precise_intensity = workbook.add_format({'num_format': '0.00000E+00', 'align': 'center'})

        for col in range(len(states_columns)):
            worksheet.set_column(0, col, 30)
            worksheet.write(0, col, states_columns[col], bold)

        for row in range(states_arr.shape[0]):
            for col in range(states_arr.shape[1]):
                if col != 3:
                    worksheet.write(row + 1, col, states_arr[row, col], format_precise_general)
                else:
                    worksheet.write(row + 1, col, states_arr[row, col], format_precise_intensity)

        workbook.close()
