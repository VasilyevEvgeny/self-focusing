from core.diffraction import FourierDiffractionExecutor_XY
from core.kerr_effect import KerrExecutor_XY
from core.propagation import Propagator
from core.beam import Beam_XY
from core.args import parse_args
from core.libs import *


args = parse_args()

beam = Beam_XY(medium="SiO2",
               distribution_type="vortex",
               P0_to_Pcr_V=5,
               m=1,
               noise_percent=0.0,
               lmbda=1800*10**-9,
               x_0=100*10**-6,
               y_0=100*10**-6,
               n_x=2048,
               n_y=2048)

propagator = Propagator(global_root_dir=args.global_root_dir,
                        beam=beam,
                        diffraction=FourierDiffractionExecutor_XY(beam=beam),
                        kerr_effect=KerrExecutor_XY(beam=beam),
                        n_z=2000,
                        dz0=10**-5,
                        flag_const_dz=True,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        plot_beam_normalization="local")

propagator.propagate()

