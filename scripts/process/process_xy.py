from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, BeamVisualizer, \
    parse_args

# parse args from command line
args = parse_args()

# create noise object
noise = GaussianNoise(r_corr=20*10**-6,
                      variance=1)

# create object of 3D beam
beam = BeamXY(medium='LiF',
              p_0_to_p_vortex=5,
              m=1,
              M=1,
              noise_percent=3.0,
              noise=noise,
              lmbda=1800*10**-9,
              x_0=100*10**-6,
              y_0=100*10**-6,
              n_x=2048,
              n_y=2048)

# create visualizer object
visualizer = BeamVisualizer(beam=beam,
                            maximum_intensity='local',
                            normalize_intensity_to=beam.i_0,
                            plot_type='volume')

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                        kerr_effect=KerrExecutorXY(beam=beam),
                        n_z=1,
                        dz_0=10**-4,
                        const_dz=False,
                        print_current_state_every=50,
                        plot_beam_every=50,
                        max_intensity_to_stop=10**17,
                        visualizer=visualizer)

# initiate propagation process
propagator.propagate()
