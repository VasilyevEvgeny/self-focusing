from core import BeamX, SweepDiffractionExecutorX, KerrExecutorX, Propagator, parse_args


args = parse_args()

beam = BeamX(medium='LiF',
             M=1,
             half=True,
             lmbda=1800*10**-9,
             x_0=92*10**-6,
             n_x=4096,
             r_kerr=50.0)

propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorX(beam=beam),
                        kerr_effect=KerrExecutorX(beam=beam),
                        n_z=500,
                        dz0=beam.z_diff/1000,
                        flag_const_dz=True,
                        dn_print_current_state=10,
                        dn_plot_beam=10,
                        beam_normalization_type='local')

propagator.propagate()
