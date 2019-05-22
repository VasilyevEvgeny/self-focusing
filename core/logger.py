from collections import OrderedDict
from time import time
from datetime import timedelta
from xlsxwriter import Workbook

from .functions import compile_to_pdf


class Logger:
    def __init__(self, **kwargs):
        self.__path = kwargs['path']
        self.__diffraction = kwargs['diffraction']
        self.__kerr_effect = kwargs['kerr_effect']

        self.__functions = OrderedDict()

    def measure_time(self, function, args):
        t_start = time()
        res = function(*args)
        t_end = time()
        duration = t_end - t_start
        function_name = function.__name__
        if function_name in self.__functions.keys():
            self.__functions[function_name] += duration
        else:
            self.__functions.update({function_name: 0.0})

        return res

    def log_times(self):
        with open(self.__path + '/times.log', 'w') as f:
            f.write('{:40s} | {:15}'.format('MODULE', 'TIME (hh:mm:ss)\n'))
            f.write('--------------------------------------------------------------\n')
            for key in self.__functions:
                f.write('{:40s} | {:10}\n'.format(key, str(timedelta(seconds=self.__functions[key]))))

    def save_initial_parameters(self, beam, n_z, dz0, max_intensity_to_stop, filename='parameters'):
        tex_file_name = filename + '.tex'
        tex_file_path = self.__path + '/' + tex_file_name

##########################
# BEGIN
##########################

#
#

        tex_file_data = \
'''\documentclass[10pt]{extarticle}

\\usepackage[left=2cm, right=2cm, top=1cm, bottom=1cm]{geometry}

\\usepackage{array}
\\newcolumntype{P}[1]{>{\centering\\arraybackslash}p{#1}}
\\newcolumntype{M}[1]{>{\centering\\arraybackslash}m{#1}}

\\usepackage[table]{xcolor}

\\usepackage{booktabs}

\\renewcommand{\\arraystretch}{1.2}
\setlength{\\tabcolsep}{0pt}

\\begin{document}
\pagestyle{empty}
\\begin{center}
\\begin{tabular}{M{5cm}M{5cm}M{5cm}}
'''

##########################
# EQUATION
##########################
        left_x_str = '2 i k_0 \\frac{\partial A(x,z)}{\partial z}'
        left_r_str = '2 i k_0 \\frac{\partial A(r,z)}{\partial z}'
        left_xy_str = '2 i k_0 \\frac{\partial A(x,y,z)}{\partial z}'

        diffraction_x_str = '\\frac{\partial^2 A(x,z)}{\partial x^2}'
        if beam.distribution_type == 'vortex':
            diffraction_r_str = '\\biggl[ \\frac{\partial^2}{\partial r^2} + \\frac1{r}\\frac{\partial}{\partial r} - \\frac{m^2}{r^2} \\biggr] A(r,z)'
        else:
            diffraction_r_str = '\\biggl[ \\frac{\partial^2}{\partial r^2} + \\frac1{r}\\frac{\partial}{\partial r} \\biggr] A(r,z)'
        diffraction_xy_str = '\\biggl[ \\frac{\partial^2}{\partial x^2} + \\frac{\partial^2}{\partial y^2} \\biggr] A(x,y,z)'

        kerr_x_str = '\\frac{2 k_0^2}{n_0} n_2 I(x,z) A(x,z)'
        kerr_r_str = '\\frac{2 k_0^2}{n_0} n_2 I(r,z) A(r,z)'
        kerr_xy_str = '\\frac{2 k_0^2}{n_0} n_2 I(x,y,z) A(x,y,z)'

        equation = []
        if beam.info == 'beam_x':
            equation.append(left_x_str)
            if self.__diffraction is not None:
                equation.append(diffraction_x_str)
            if self.__kerr_effect is not None:
                equation.append(kerr_x_str)
        elif beam.info == 'beam_r':
            equation.append(left_r_str)
            if self.__diffraction is not None:
                equation.append(diffraction_r_str)
            if self.__kerr_effect is not None:
                equation.append(kerr_r_str)
        elif beam.info == 'beam_xy':
            equation.append(left_xy_str)
            if self.__diffraction is not None:
                equation.append(diffraction_xy_str)
            if self.__kerr_effect is not None:
                equation.append(kerr_xy_str)

        eq_length = len(equation)
        if eq_length == 1:
            equation.append('= 0')
        elif eq_length == 2:
            equation = equation[:1] + ['='] + equation[1:]
        elif eq_length == 3:
            equation = equation[:1] + ['='] + equation[1:2] + ['+'] + equation[2:]

        equation_str = ' '.join(equation)

        tex_file_data += \
'''\\midrule[2pt]
\multicolumn{3}{M{15cm}}{\\textbf{EQUATION}} \\tabularnewline
\\midrule[2pt]
\multicolumn{3}{M{15cm}}{\[ %s \]} \\tabularnewline
\\midrule[2pt]
''' % equation_str

##########################
# INITIAL CONDITION
##########################

        initial_condition_str = None
        if beam.info == 'beam_x':
            initial_condition_str = 'A(x,0) = A_0 \\biggl( \\frac{x}{x_0} \\biggr)^M \exp \\biggl\{ -\\frac{x^2}{2x_0^2} \\biggr\}'
        elif beam.info == 'beam_r':
            initial_condition_str = 'A(r,0) = A_0 \\biggl( \\frac{r}{r_0} \\biggr)^M \exp \\biggl\{ -\\frac{r^2}{2r_0^2} \\biggr\}'
        elif beam.info == 'beam_xy':
            initial_condition_str = 'A(x,y,0) = \\biggl(1 + C \\xi(x,y)\\biggr)A_0 \\biggl(\\frac{x^2}{x_0^2}+\\frac{y^2}{y_0^2}\\biggr)^{M/2}\exp\\biggl\{-\\frac1{2}\\biggl(\\frac{x^2}{x_0^2}+\\frac{y^2}{y_0^2}\\biggr)\\biggr\}\exp\\biggl\{i m \\varphi(x,y)\\biggr\}'

        tex_file_data += \
'''\multicolumn{3}{M{15cm}}{\\textbf{INITIAL CONDITION}} \\tabularnewline
\\midrule[2pt]
\multicolumn{3}{M{15cm}}{\[ %s \]} \\tabularnewline
\\midrule[2pt]
''' % initial_condition_str


    ##########################
    # MEDIUM
    ##########################

        if beam.medium.info == 'SiO2':
            material = 'SiO$_2$'
        elif beam.medium.info == 'CaF2':
            material = 'CaF$_2$'
        elif beam.medium.info == 'LiF':
            material = 'LiF'
        else:
            material = 'noname'

        tex_file_data += \
'''\multicolumn{3}{M{15cm}}{\\textbf{MEDIUM}} \\tabularnewline
\\midrule[2pt]
material & %s & -- \\tabularnewline
\hline
$n_0$ & %1.4f & -- \\tabularnewline
\hline
$n_2$ & %1.2f $\\times 10^{-16}$ & cm$^2$/W \\tabularnewline
\hline
$k_0$ & %.2f & 1/mm \\tabularnewline
\hline
$k_1$ & %.2f & fs/mm \\tabularnewline
\hline
$k_2$ & %.2f & fs$^2$/mm \\tabularnewline
\\midrule[2pt]
''' % (material, beam.medium.n_0, beam.medium.n_2 * 10**20, beam.medium.k_0 * 10**-3, beam.medium.k_1 * 10**12, beam.medium.k_2 * 10**27)


##########################
# BEAM
##########################

        tex_file_data += \
'''\multicolumn{3}{M{15cm}}{\\textbf{BEAM}} \\tabularnewline
\\midrule[2pt]
distribution & %s & -- \\tabularnewline
\hline
'''% (beam.distribution_type)

        tex_file_data += \
'''$M$ & %d & -- \\tabularnewline
\hline
''' % (beam.M)

        if beam.info in ('beam_r', 'beam_xy'):
            tex_file_data += \
'''$m$ & %d & -- \\tabularnewline
\hline
''' % (beam.m)

        if beam.info == 'beam_x':
            tex_file_data += \
'''$x_0$ & %d & $\mu$m \\tabularnewline
\hline
''' % (round(beam.x_0 * 10 ** 6))
        elif beam.info == 'beam_r':
            tex_file_data += \
'''$r_0$ & %d & $\mu$m \\tabularnewline
\hline
''' % (round(beam.r_0 * 10**6))
        elif beam.info == 'beam_xy':
            tex_file_data += \
'''$x_0$ & %d & $\mu$m \\tabularnewline
\hline
$y_0$ & %d & $\mu$m \\tabularnewline
\hline
''' % (round(beam.x_0 * 10 ** 6), round(beam.y_0 * 10 ** 6))

        tex_file_data += \
'''
$\lambda$ & %d & nm \\tabularnewline
\hline
$z_{diff}$ & %2.4f & cm \\tabularnewline
\hline
''' % (beam.lmbda * 10**9, beam.z_diff * 10**2)

        if beam.info in ('beam_r', 'beam_xy'):
            if beam.distribution_type == 'gauss' or beam.distribution_type == 'ring':
                tex_file_data += \
'''$P_0 / P_G$ & %.2f & -- \\tabularnewline
\hline
''' % (beam.p_0_to_p_gauss)

            if beam.distribution_type == 'vortex':
                tex_file_data += \
'''$P_0 / P_V$ & %.2f & -- \\tabularnewline
\hline
''' % (beam.p_0_to_p_vortex)

            if beam.info in ('beam_r', 'beam_xy'):
                tex_file_data += \
'''$P_0$ & %.2f & MW \\tabularnewline
\hline
''' % (beam.p_0 * 10**-6)

        tex_file_data += \
'''$I_0$ & %.4f & TW/cm$^2$ \\tabularnewline
\hline
$R_{kerr}$ & %.2f & -- \\tabularnewline
''' % (beam.i_0 * 10**-16, beam.r_kerr)

        if beam.info == 'beam_xy':
            tex_file_data += \
'''\hline
C & %.2f & -- \\tabularnewline
''' % (beam.noise_percent / 100)

        tex_file_data += \
'''\\midrule[2pt]'''

##########################
# GRID
##########################

        tex_file_data += \
'''\multicolumn{3}{M{15cm}}{\\textbf{GRID}} \\tabularnewline
\\midrule[2pt]
'''

        if beam.info == 'beam_x':
            tex_file_data += \
'''$x_{max}$ & % d & $\mu$m \\tabularnewline
\hline
$n_x$ & %d & -- \\tabularnewline
\hline
$h_x$ & %.2f & $\mu$m \\tabularnewline
''' % (round(beam.x_max * 10 ** 6), beam.n_x, round(beam.dx * 10 ** 6))
        elif beam.info == 'beam_r':
            tex_file_data += \
'''$r_{max}$ & % d & $\mu$m \\tabularnewline
\hline
$n_r$ & %d & -- \\tabularnewline
\hline
$h_r$ & %.2f & $\mu$m \\tabularnewline
''' % (round(beam.r_max * 10**6), beam.n_r, round(beam.dr * 10**6))
        elif beam.info == 'beam_xy':
            tex_file_data += \
'''$x_{max}$ & %d & $\mu$m \\tabularnewline
\hline
$y_{max}$ & %d & $\mu$m \\tabularnewline
\hline
$n_x$ & %d & -- \\tabularnewline
\hline
$n_y$ & %d & -- \\tabularnewline
\hline
$h_x$ & %.2f & $\mu$m \\tabularnewline
\hline
$h_y$ & %.2f & $\mu$m \\tabularnewline
''' % (round(beam.x_max * 10 ** 6), round(beam.y_max * 10 ** 6),
       beam.n_x, beam.n_y,
       round(beam.dx * 10 ** 6),
       round(beam.dy * 10 ** 6))

        tex_file_data += \
'''\\midrule[2pt]'''

##########################
# TRACK
##########################

        tex_file_data += \
'''\multicolumn{3}{M{15cm}}{\\textbf{TRACK}} \\tabularnewline
\\midrule[2pt]
$n_z$ & %d & -- \\tabularnewline
\hline
$h_z (z=0)$ & %.2f & $\mu$m \\tabularnewline
\hline
$I_{stop}$ & %.2f & TW/cm$^2$ \\tabularnewline
\hline
''' % (n_z, dz0 * 10**6, max_intensity_to_stop * 10**-16)

##########################
# END
##########################

        tex_file_data += \
'''\end{tabular}
\end{center}
\end{document}
'''

        with open(tex_file_path, 'w') as f:
            f.write(tex_file_data)

        compile_to_pdf(tex_file_path, delete_tex_file=True)

    @staticmethod
    def print_current_state(n_step, states, states_columns):
        if n_step == 0:
            print('      |   %s   |    %s   |  %s |  %s |' % (states_columns[0],
                                                         states_columns[1],
                                                         states_columns[2],
                                                         states_columns[3]))
        output_string = '{:04d} {:11.6f} {:13e} {:11.6f} {:17e}'.format(n_step,
                                                                      states[n_step, 0],
                                                                      states[n_step, 1],
                                                                      states[n_step, 2],
                                                                      states[n_step, 3])
        print(output_string)

    def log_track(self, states_arr, states_columns, filename='propagation.xlsx'):
        filename = self.__path + '/' + filename

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
