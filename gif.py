from core.diffraction import FourierDiffractionExecutor_XY
from core.kerr_effect import KerrExecutor_XY
from core.propagation import Propagator
from core.beam import Beam_XY
from core.args import parse_args
from core.functions import get_files_for_gif, make_animations
from core.libs import *

args = parse_args()

all_files = []
names = []
n_pictures_max = 0
for noise_percent in [20]:
    for m in [1, 2]:

        beam = Beam_XY(medium="SiO2",
                       distribution_type="vortex",
                       P0_to_Pcr_V=5,
                       m=m,
                       noise_percent=noise_percent,
                       lmbda=1800*10**-9,
                       x_0=100*10**-6,
                       y_0=100*10**-6,
                       n_x=256,
                       n_y=256)

        propagator = Propagator(args=args,
                                beam=beam,
                                diffraction=FourierDiffractionExecutor_XY(beam=beam),
                                kerr_effect=KerrExecutor_XY(beam=beam),
                                n_z=2000,
                                dz0=10**-5,
                                flag_const_dz=True,
                                dn_print_current_state=50,
                                dn_plot_beam=10,
                                plot_beam_normalization="local")

        propagator.propagate()

        files, n_pictures = get_files_for_gif(args.global_root_dir + "/" + args.global_results_dir_name)
        all_files.append(files)

        if n_pictures > n_pictures_max:
            n_pictures_max = n_pictures

        name = "noise_percent=%02d" % noise_percent + "__m=%d" % m
        names.append(name)

        #print(n_pictures)
        #print(files)

        del beam
        del propagator


make_animations(all_files, names, n_pictures_max)