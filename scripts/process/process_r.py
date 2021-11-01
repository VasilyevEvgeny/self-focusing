from core import BeamR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, BeamVisualizer, \
    parse_args

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamR(medium='SiO2',
             p_0_to_p_vortex=9,
             m=4,
             M=4,
             lmbda=1800*10**-9,
             r_0=100*10**-6,
             n_r=1024)

# create visualizer object
visualizer = BeamVisualizer(beam=beam,
                            maximum_intensity='local',
                            normalize_intensity_to=1,
                            plot_type='flat')

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=SweepDiffractionExecutorR(beam=beam),
                        kerr_effect=KerrExecutorR(beam=beam),
                        n_z=5000,
                        dz_0=beam.z_diff / 1000,
                        const_dz=False,
                        print_current_state_every=1,
                        plot_beam_every=1000,
                        max_intensity_to_stop=5e17,
                        visualizer=visualizer)

# initiate propagation process
propagator.propagate()
