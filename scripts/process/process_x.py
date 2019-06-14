from core import BeamX, SweepDiffractionExecutorX, KerrExecutorX, Propagator, BeamVisualizator2D, parse_args

# parse args from command line
args = parse_args()

# create object of 2D beam
beam = BeamX(medium='LiF',
             M=1,
             half=True,
             lmbda=1557*10**-9,
             x_0=85*10**-6,
             n_x=4096,
             r_kerr=75.4)

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorX(beam=beam),
                        kerr_effect=KerrExecutorX(beam=beam),
                        n_z=1000,
                        dz_0=beam.z_diff/1000,
                        const_dz=True,
                        print_current_state_every=10,
                        plot_beam_every=1,
                        plot_beam_maximum='local',
                        plot_beam_func=BeamVisualizator2D.plot_beam_profile)

# initiate propagation process
propagator.propagate()
