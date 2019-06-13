from core import BeamXY, GaussianNoise, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, parse_args

# parse args from command line
args = parse_args()

# create noise object
noise = GaussianNoise(r_corr_in_meters=50*10**-6,
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
                        dz0=10**-4,
                        flag_const_dz=False,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        beam_normalization_type='local',
                        beam_in_3D=False)

# initiate propagation process
propagator.propagate()
