from core import BeamX, SweepDiffractionExecutorX, KerrExecutorX, Propagator, BeamVisualizer, parse_args

# parse args from command line
args = parse_args()

# create object of 2D beam
beam = BeamX(medium='LiF',
             M=1,
             half=True,
             lmbda=1800*10**-9,
             x_0=92*10**-6,
             n_x=1024,
             r_kerr=75.4)

# create visualizer object
visualizer = BeamVisualizer(beam=beam,
                            maximum_intensity='local',
                            normalize_intensity_to=beam.i_0,
                            plot_type='profile')

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorX(beam=beam),
                        kerr_effect=KerrExecutorX(beam=beam),
                        n_z=500,
                        dz_0=beam.z_diff/1000,
                        const_dz=True,
                        print_current_state_every=50,
                        plot_beam_every=50,
                        save_field=True,
                        visualizer=visualizer)

# initiate propagation process
propagator.propagate()
