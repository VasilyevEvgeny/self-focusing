from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, BeamVisualizator3D, parse_args

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamR(medium='LiF',
             p_0_to_p_vortex=5,
             m=1,
             M=1,
             lmbda=1557*10**-9,
             r_0=85*10**-6,
             n_r=512)

visualizator = BeamVisualizator3D(beam=beam)

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorR(beam=beam),
                        kerr_effect=KerrExecutorR(beam=beam),
                        n_z=1000,
                        dz_0=beam.z_diff / 1000,
                        const_dz=True,
                        print_current_state_every=50,
                        plot_beam_every=50,
                        plot_beam_maximum=3,
                        plot_beam_func=BeamVisualizator3D.plot_beam_flat)

# initiate propagation process
propagator.propagate()
