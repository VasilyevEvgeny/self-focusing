from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, BeamVisualizer, \
    parse_args

# parse args from command line
args = parse_args()

# create noise object
noise = GaussianNoise(r_corr_in_meters=1e-6,
                      variance=1)

# create object of 3D beam
beam = BeamXY(medium='LiF',
              p_0_to_p_vortex=8,
              m=1,
              M=1,
              noise_percent=0.001,
              noise=noise,
              lmbda=1800*10**-9,
              x_0=100*10**-6,
              y_0=100*10**-6,
              n_x=4096,
              n_y=4096)

# create visualizer object
visualizer = BeamVisualizer(beam=beam,
                            maximum_intensity='local',
                            normalize_intensity_to=1,
                            plot_type='flat')

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                        kerr_effect=KerrExecutorXY(beam=beam),
                        n_z=1,
                        dz_0=10**-4,
                        const_dz=False,
                        print_current_state_every=5,
                        plot_beam_every=5,
                        max_intensity_to_stop=5*10**17,
                        visualizer=visualizer,
                        save_field=False)

# initiate propagation process
propagator.propagate()
