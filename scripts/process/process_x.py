from core import BeamX, SweepDiffractionExecutorX, KerrExecutorX, Propagator, parse_args


args = parse_args()

beam = BeamX(medium='LiF',
             M=1,
             half=True,
             lmbda=1557*10**-9,
             x_0=85*10**-6,
             n_x=4096,
             r_kerr=75.4)

propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorX(beam=beam),
                        kerr_effect=KerrExecutorX(beam=beam),
                        n_z=1,
                        dz0=beam.z_diff/1000,
                        flag_const_dz=True,
                        dn_print_current_state=10,
                        dn_plot_beam=1,
                        beam_normalization_type='local')

propagator.propagate()
