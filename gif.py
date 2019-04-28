from core.diffraction import FourierDiffractionExecutor_XY
from core.kerr_effect import KerrExecutor_XY
from core.propagation import Propagator
from core.beam import Beam_XY
from core.args import parse_args
from core.functions import get_files_for_gif, make_animation
from core.libs import *

args = parse_args()

for noise_percent in [0, 10, 20]:
    for m in [1, 2]:

        beam = Beam_XY(medium="SiO2",
                       distribution_type="vortex",
                       P0_to_Pcr_V=5,
                       m=m,
                       noise_percent=noise_percent,
                       lmbda=1800*10**-9,
                       x_0=100*10**-6,
                       y_0=100*10**-6,
                       n_x=512,
                       n_y=512)


        propagator = Propagator(args=args,
                                beam=beam,
                                diffraction=FourierDiffractionExecutor_XY(beam=beam),
                                kerr_effect=KerrExecutor_XY(beam=beam),
                                n_z=1000,
                                dz0=10**-5,
                                flag_const_dz=True,
                                dn_print_current_state=50,
                                dn_plot_beam=10,
                                plot_beam_normalization="local")

        propagator.propagate()

        files = get_files_for_gif(args.global_root_dir + "/" + args.global_results_dir_name)

        name = "noise_percent=%02d" % noise_percent + "__m=%d" % m
        make_animation(files, name)

        del beam
        del propagator