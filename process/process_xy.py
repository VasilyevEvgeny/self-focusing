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
               n_x=512,
               n_y=512)

propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=FourierDiffractionExecutor_XY(beam=beam),
                        kerr_effect=KerrExecutor_XY(beam=beam),
                        n_z=5000,
                        dz0=10**-5,
                        flag_const_dz=False,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        beam_normalization_type="local")

propagator.propagate()

