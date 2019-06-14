from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, BeamVisualizator3D, \
    parse_args

# parse args from command line
args = parse_args()

# create noise object
noise = GaussianNoise(r_corr=50*10**-6,
                      variance=1)

# create object of 3D beam
beam = BeamXY(medium='LiF',
              p_0_to_p_vortex=5,
              m=0,
              M=0,
              noise_percent=3.0,
              noise=noise,
              lmbda=1800*10**-9,
              x_0=100*10**-6,
              y_0=100*10**-6,
              n_x=512,
              n_y=512)

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
                        plot_beam_maximum='local',
                        plot_beam_func=BeamVisualizator3D.plot_beam_flat)

# initiate propagation process
propagator.propagate()
