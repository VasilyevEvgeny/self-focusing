from core.diffraction import SweepDiffractionExecutor_R
from core.kerr_effect import KerrExecutor_R
from core.propagation import Propagator
from core.beam import Beam_R
from core.args import parse_args
from core.libs import *


args = parse_args()

beam = Beam_R(medium="SiO2",
              distribution_type="vortex",
              P0_to_Pcr_V=5,
              m=1,
              amp_noise_coeff=0.00,
              phase_noise_coeff=0.00,
              lmbda=1800*10**-9,
              r_0=100*10**-6,
              n_r=1024)

propagator = Propagator(global_root_dir=args.global_root_dir,
                        beam=beam,
                        diffraction=SweepDiffractionExecutor_R(beam=beam),
                        #kerr_effect=KerrExecutor_R(beam=beam),
                        n_z=1000,
                        dz0=beam.z_diff / 1000,
                        flag_const_dz=True,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        plot_beam_normalization="local")

propagator.propagate()
