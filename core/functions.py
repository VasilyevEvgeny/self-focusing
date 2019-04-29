from core.libs import *


def calc_ticks_x(labels, xs):
    ticks = []
    nxt = 0
    for label in labels:
        for i in range(nxt, len(xs)):
            if xs[i] > float(label) * 10**-6:
                ticks.append(i)
                nxt = i
                break
    return ticks


def crop_x(arr, xs, x_left, x_right, mode):
    i_min, i_max = 0, -1
    for i in range(len(xs) - 1, 0, -1):
        if xs[i] < x_left:
            i_min = i
            break
    for i in range(len(xs)):
        if xs[i] > x_right:
            i_max = i
            break
    if mode == "x":
        return np.transpose(np.transpose(arr)[i_min:i_max, :]), i_min, i_max
    elif mode == "y":
        return np.transpose(np.transpose(arr)[:, i_min:i_max]), i_min, i_max
    else:
        raise Exception('Wrong mode in crop_x!')


@jit(nopython=True)
def linear_approximation_complex(x, x1, y1, x2, y2):
    return complex((y1.real - y2.real) / (x1 - x2) * x + (y2.real * x1 - x2 * y1.real) / (x1 - x2),\
                   (y1.imag - y2.imag) / (x1 - x2) * x + (y2.imag * x1 - x2 * y1.imag) / (x1 - x2))


@jit(nopython=True)
def linear_approximation_real(x, x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2) * x + (y2 * x1 - x2 * y1) / (x1 - x2)


@jit(nopython=True)
def r_to_xy_real(r_slice):
    n_r = len(r_slice)
    n_x, n_y = 2 * n_r, 2 * n_r
    arr = np.zeros(shape=(n_x, n_y), dtype=np.float64)
    for i in range(n_x):
        for j in range(n_y):
            r = sqrt((i - n_x / 2.) ** 2 + (j - n_y / 2.) ** 2)
            if int(r) < n_r - 1:
                arr[i, j] = linear_approximation_real(r, int(r), r_slice[int(r)], int(r) + 1, r_slice[int(r) + 1])
    return arr


def get_files_for_gif(path, prefix="GIF_"):
    paths_with_gifs = []
    for path in glob(path + "/*"):
        if path.split("\\")[-1][:4] == prefix:
            paths_with_gifs.append(path)

    files = []
    n_pictures = 0
    for file in glob(paths_with_gifs[-1] + "/beam/*"):
        files.append(file.replace("\\", "/"))
        n_pictures += 1

    return files, n_pictures


def make_animations(all_files, names, n_pictures_max, path="./gifs", fps=10):

    for idx in range(len(all_files)):
        images = []
        for file in all_files[idx]:
            images.append(imageio.imread(file))

        delta = n_pictures_max - len(images)
        for i in range(delta):
            images.append(imageio.imread(all_files[idx][-1]))

        # 1 second pause at the beginning
        for i in range(fps):
            images = [imageio.imread(all_files[idx][0])] + images

        # 1 second pause at the end
        for i in range(fps):
            images.append(imageio.imread(all_files[idx][-1]))

        print(len(images))

        imageio.mimsave(path + "/" + names[idx] + ".gif", images, fps=fps)