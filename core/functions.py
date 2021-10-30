from numpy import sqrt, transpose, zeros, float64, complex64, pi
from scipy.special import gamma
from numba import jit
from glob import glob
from datetime import datetime
import os
import shutil
from time import sleep
import imageio
import cv2
import subprocess
import pandas as pd
import argparse
import pathlib


def load_dirnames(path=os.getcwd() + '/tests/dirnames.txt'):

    if pathlib.Path(path).exists():
        with open(path, 'r') as f:
            global_root_dir = f.readline()
            global_results_dir_name = f.readline()

            if global_root_dir[-1] == '\n':
                global_root_dir = global_root_dir[:-1]

            if global_results_dir_name[-1] == '\n':
                global_results_dir_name = global_results_dir_name[:-1]

            return global_root_dir, global_results_dir_name
    else:
        raise Exception('No file with dirnames in test mode!')


def parse_args():
    """Parses arguments from command line"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--global_root_dir')
    parser.add_argument('--global_results_dir_name')
    parser.add_argument('--prefix')
    parser.add_argument('--insert_datetime', default=True)

    return parser.parse_args()


def xlsx_to_df(path_to_xlsx, normalize_z_to=10**2, normalize_i_to=10**17):
    """Converts xlsx propagation file to pandas dataframe with some normalized columns"""

    df = pd.read_excel(path_to_xlsx)

    df['z, m'] *= normalize_z_to
    df['dz, m'] *= normalize_z_to
    df['i_max, W / m^2'] /= normalize_i_to

    df = df.rename(index=str, columns={'z, m': 'z_normalized', 'dz, m': 'dz_normalized', 'i_max, W / m^2': 'i_max_normalized'})

    return df


def calc_ticks_x(labels, xs):
    """Calculates grid points corresponding to labels along axis"""

    ticks = []
    nxt = 0
    for label in labels:
        for i in range(nxt, len(xs)):
            if xs[i] > float(label.replace('$-$', '-')) * 10**-6:
                ticks.append(i)
                nxt = i
                break
    return ticks


def crop_x(arr, xs, x_left, x_right, mode):
    """Crops array along axis x from x_left to x_right"""

    i_min, i_max = 0, -1
    for i in range(len(xs) - 1, 0, -1):
        if xs[i] < x_left:
            i_min = i
            break
    for i in range(len(xs)):
        if xs[i] > x_right:
            i_max = i
            break
    if mode == 'x':
        return transpose(transpose(arr)[i_min:i_max, :]), i_min, i_max
    elif mode == 'y':
        return transpose(transpose(arr)[:, i_min:i_max]), i_min, i_max
    else:
        raise Exception('Wrong mode in crop_x!')


@jit(nopython=True)
def linear_approximation_complex(x, x1, y1, x2, y2):
    """Linear approximation for complex arguments"""

    return complex((y1.real - y2.real) / (x1 - x2) * x + (y2.real * x1 - x2 * y1.real) / (x1 - x2),
                   (y1.imag - y2.imag) / (x1 - x2) * x + (y2.imag * x1 - x2 * y1.imag) / (x1 - x2))


@jit(nopython=True)
def linear_approximation_real(x, x1, y1, x2, y2):
    """Linear approximation for float arguments"""

    return (y1 - y2) / (x1 - x2) * x + (y2 * x1 - x2 * y1) / (x1 - x2)


@jit(nopython=True)
def r_to_xy_real(r_slice):
    """Converts 1D array with data along radius-vector r to 2D array with axially symmetric data (x,y)"""

    n_r = len(r_slice)
    n_x, n_y = 2 * n_r, 2 * n_r
    arr = zeros(shape=(n_x, n_y), dtype=float64)
    for i in range(n_x):
        for j in range(n_y):
            r = sqrt((i - n_x / 2.) ** 2 + (j - n_y / 2.) ** 2)
            if int(r) < n_r - 1:
                arr[i, j] = linear_approximation_real(r, int(r), r_slice[int(r)], int(r) + 1, r_slice[int(r) + 1])
    return arr


@jit(nopython=True)
def r_to_xy_complex(r_slice):
    """Converts 1D array with data along radius-vector r to 2D array with axially symmetric data (x,y)"""

    n_r = len(r_slice)
    n_x, n_y = 2 * n_r, 2 * n_r
    arr = zeros(shape=(n_x, n_y), dtype=complex64)
    for i in range(n_x):
        for j in range(n_y):
            r = sqrt((i - n_x / 2.) ** 2 + (j - n_y / 2.) ** 2)
            if int(r) < n_r - 1:
                arr[i, j] = linear_approximation_complex(r, int(r), r_slice[int(r)], int(r) + 1, r_slice[int(r) + 1])
    return arr


def make_paths(global_root_dir, global_results_dir_name, prefix, insert_datetime=True):
    """Returns paths formed from command line arguments"""

    global_results_dir = global_root_dir + '/' + global_results_dir_name

    if insert_datetime:
        datetime_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        datetime_string = ''

    if prefix is None:
        results_dir_name = datetime_string
    else:
        if datetime_string:
            results_dir_name = prefix + '_' + datetime_string
        else:
            results_dir_name = prefix

    results_dir = global_results_dir + '/' + results_dir_name

    return global_results_dir, results_dir, results_dir_name


def create_dir(**kwargs):
    """Creates dir with default name 'images' deleting the existing directory"""

    path = kwargs['path']
    dir_name = kwargs.get('dir_name', 'images')

    res_path = path + '/' + dir_name

    try:
        os.makedirs(res_path)
    except OSError:
        shutil.rmtree(res_path)
        sleep(1)
        os.makedirs(res_path)

    return res_path


def create_multidir(global_root_dir, global_results_dir_name, prefix):
    """Creates directory for multidir mode"""

    global_results_dir, results_dir, results_dir_name = make_paths(global_root_dir, global_results_dir_name, prefix)
    create_dir(path=global_results_dir, dir_name=results_dir_name)

    return results_dir, results_dir_name


def make_animation(root_dir, name, images_dir_name='images', fps=10):
    """Makes gif-animation from series of pictures"""

    images_for_animation = []
    for file in glob(root_dir + '/' + images_dir_name + '/*.png'):
        images_for_animation.append(imageio.imread(file))
    imageio.mimsave(root_dir + '/' + name + '.gif', images_for_animation, fps=fps)


def make_video(root_dir, name, images_dir_name='images', fps=10):
    """Makes video from series of pictures"""

    images_for_video = []
    for file in glob(root_dir + '/' + images_dir_name + '/*.png'):
        images_for_video.append(cv2.imread(file))

    height, width, leyers = images_for_video[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(root_dir + '/' + name + '.avi', fourcc, fps, (width, height))

    for img in images_for_video:
        video.write(cv2.resize(img, (width, height)))

    cv2.destroyAllWindows()
    video.release()


def compile_to_pdf(tex_file_path, delete_tmp_files=True, delete_tex_file=False):
    """Compiles tex-code with pdf-latex and produces pdf-file with ability to delete temporary files"""

    path_list = (tex_file_path.replace('\\', '/')).split('/')
    path, filename = '/'.join(path_list[:-1]), path_list[-1].split('.')[0]

    try:
        subprocess.check_output(
            ['pdflatex', '-quiet', '-interaction=nonstopmode', tex_file_path, '-output-directory', path])
    except:
        Exception('Wrong pdflatex compilation!')

    if delete_tmp_files:
        for ext in ['aux', 'log', 'out', 'fls', 'fdb_latexmk', 'dvi']:
            try:
                file = path + '/' + filename + '.' + ext
                os.remove(file)
            except:
                pass

    if delete_tex_file:
        try:
            os.remove(path + '/' + filename + '.tex')
        except:
            pass


def calculate_p_gauss(lmbda, n_0, n_2):
    """Calculates critical power of self-focusing for Gaussian beam"""

    return 3.77 * lmbda ** 2 / (8 * pi * n_0 * n_2)


def calculate_p_vortex(m, p_gauss):
    """Calculates critical power of self-focusing for vortex beam"""

    return p_gauss * 2**(2 * m + 1) * gamma(m + 1) * gamma(m + 2) / (2 * gamma(2 * m + 1))
