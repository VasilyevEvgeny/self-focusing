from core import Beam_R, SweepDiffractionExecutor_R, KerrExecutor_R, Propagator, parse_args


args = parse_args()

beam = Beam_R(medium="SiO2",
              P0_to_Pcr_G=5,
              m=0,
              M=0,
              lmbda=1800*10**-9,
              r_0=100*10**-6,
              n_r=4096)

propagator = Propagator(args=args,
                        beam=beam,
                        #diffraction=SweepDiffractionExecutor_R(beam=beam),
                        #kerr_effect=KerrExecutor_R(beam=beam),
                        n_z=1,
                        dz0=10**-5,
                        flag_const_dz=True,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        beam_normalization_type="local")

propagator.propagate()
