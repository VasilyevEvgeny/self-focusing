from core import Beam_XY, GaussianNoise, FourierDiffractionExecutor_XY, KerrExecutor_XY, Propagator, parse_args


args = parse_args()

noise = GaussianNoise(r_corr_in_meters=30*10**-6,
                      variance=1)

beam = Beam_XY(medium='SiO2',
               P0_to_Pcr_G=5,
               m=0,
               M=0,
               noise_percent=1.0,
               noise=noise,
               lmbda=1800*10**-9,
               x_0=100*10**-6,
               y_0=100*10**-6,
               n_x=512,
               n_y=512)

propagator = Propagator(args=args,
                        beam=beam,
                        diffraction=FourierDiffractionExecutor_XY(beam=beam),
                        kerr_effect=KerrExecutor_XY(beam=beam),
                        n_z=1,
                        dz0=10**-5,
                        flag_const_dz=False,
                        dn_print_current_state=50,
                        dn_plot_beam=50,
                        beam_normalization_type="local")

propagator.propagate()
