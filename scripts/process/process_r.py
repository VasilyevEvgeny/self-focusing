from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, BeamVisualizer, \
    parse_args

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamR(medium='LiF',
             p_0_to_p_vortex=5,
             m=1,
             M=1,
             lmbda=1800*10**-9,
             r_0=100*10**-6,
             n_r=2048)

# create visualizer object
visualizer = BeamVisualizer(beam=beam,
                            maximum_intensity='local',
                            normalize_intensity_to=beam.i_0,
                            plot_type='volume')

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorR(beam=beam),
                        kerr_effect=KerrExecutorR(beam=beam),
                        n_z=1,
                        dz_0=beam.z_diff / 1000,
                        const_dz=True,
                        print_current_state_every=50,
                        plot_beam_every=50,
                        max_intensity_to_stop=10**17,
                        visualizer=visualizer)

# initiate propagation process
propagator.propagate()
