from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, parse_args


args = parse_args()

beam = BeamR(medium='SiO2',
             p_0_to_p_gauss=5,
             m=0,
             M=0,
             lmbda=1800*10**-9,
             r_0=100*10**-6,
             n_r=4096)

propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorR(beam=beam),
                        #kerr_effect=KerrExecutorR(beam=beam),
                        n_z=1000,
                        dz0=beam.z_diff / 1000,#10**-5,
                        flag_const_dz=True,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        beam_normalization_type='local')

propagator.propagate()
