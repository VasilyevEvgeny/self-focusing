from core.diffraction import FourierDiffractionExecutor_XY
from core.kerr_effect import KerrExecutor_XY
from core.propagation import Propagator
from core.beam import Beam_XY
from core.args import parse_args
from core.functions import create_dir, create_multidir, get_files, make_animation, make_video
from core.libs import *


def get_data():
    args = parse_args()
    results_dir, results_dir_name = create_multidir(args.global_root_dir, args.global_results_dir_name, args.prefix)

    indices = []
    for idx_noise_percent, noise_percent in enumerate([0, 10, 20]):
        for idx_m, m in enumerate([1, 2]):

            print("================================")
            print("noise_percent = %02d" % noise_percent, ", m = %d" % m)
            print("================================")

            beam = Beam_XY(medium="SiO2",
                           distribution_type="vortex",
                           P0_to_Pcr_V=5,
                           m=m,
                           M=m,
                           noise_percent=noise_percent,
                           lmbda=1800*10**-9,
                           x_0=100*10**-6,
                           y_0=100*10**-6,
                           n_x=512,
                           n_y=512)

            propagator = Propagator(args=args,
                                    multidir_name=results_dir_name,
                                    beam=beam,
                                    diffraction=FourierDiffractionExecutor_XY(beam=beam),
                                    kerr_effect=KerrExecutor_XY(beam=beam),
                                    n_z=500,
                                    dz0=10**-5,
                                    flag_const_dz=True,
                                    dn_print_current_state=50,
                                    dn_plot_beam=100,
                                    beam_normalization_type="local")

            propagator.propagate()

            del beam
            del propagator

            index = (idx_noise_percent, idx_m)
            indices.append(index)

    all_files, n_pictures_max = get_files(results_dir)

    return all_files, n_pictures_max, indices, results_dir, args.prefix


def process_multimedia(all_files, indices, n_pictures_max, path, prefix, fps=10, animation=True, video=True):
    all_files_upd = []
    for idx in range(len(all_files)):
        files = []
        for file in all_files[idx]:
            files.append(file)

        # append last picture if n_pictures < n_pictures_max
        delta = n_pictures_max - len(files)
        for i in range(delta):
            files.append(all_files[idx][-1])

        # 1 second pause at the beginning
        for i in range(fps):
            files = [all_files[idx][0]] + files

        # 1 second pause at the end
        for i in range(fps):
            files.append(all_files[idx][-1])

        all_files_upd.append(files)

    # save composed images to dir
    results_dir = create_dir(path=path)
    width, height = Image.open(all_files_upd[0][0]).size
    i1_max, i2_max = indices[-1]
    total_width, total_height = (i1_max + 1) * width, (i2_max + 1) * height
    for i in range(len(all_files_upd[0])):
        composed_im = Image.new('RGB', (total_width, total_height))
        for j in range(len(all_files_upd)):
            im = Image.open(all_files_upd[j][i])
            i1, i2 = indices[j]
            composed_im.paste(im, (i1 * width, i2 * height))
        composed_im.save(results_dir + "/%04d.png" % i, "PNG", transparent=True)

    if animation:
        make_animation(root_dir=path,
                       name=prefix,
                       fps=fps)

    if video:
        make_video(root_dir=path,
                   name=prefix,
                   fps=fps)


all_files, n_pictures_max, indices, results_dir, prefix = get_data()
process_multimedia(all_files, indices, n_pictures_max, results_dir, prefix)
