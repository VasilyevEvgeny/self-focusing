from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, parse_args

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamR(medium='LiF',
             p_0_to_p_vortex=5,
             m=1,
             M=1,
             lmbda=1557*10**-9,
             r_0=85*10**-6,
             n_r=4096)

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorR(beam=beam),
                        kerr_effect=KerrExecutorR(beam=beam),
                        n_z=1,
                        dz0=beam.z_diff / 1000,
                        flag_const_dz=True,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        beam_normalization_type=3,
                        beam_in_3D=True)

# initiate propagation process
propagator.propagate()
