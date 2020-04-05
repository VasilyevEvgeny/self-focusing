from core import BeamXY, SpectrumXY, FourierDiffractionExecutorXY, KerrExecutorXY, Propagator, SpectrumVisualizer, \
    parse_args

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamXY(medium='LiF',
              p_0_to_p_vortex=5,
              m=1,
              M=1,
              lmbda=1800*10**-9,
              x_0=100*10**-6,
              y_0=100*10**-6,
              radii_in_grid=70,
              n_x=8192,
              n_y=8192)

spectrum = SpectrumXY(beam=beam)

# create visualizer object
spectrum_visualizer = SpectrumVisualizer(spectrum=spectrum,
                                         remaining_central_part_coeff_field=0.12,
                                         remaining_central_part_coeff_spectrum=0.02)

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        spectrum=spectrum,
                        diffraction=FourierDiffractionExecutorXY(beam=beam),
                        kerr_effect=KerrExecutorXY(beam=beam),
                        n_z=200,
                        dz_0=beam.z_diff / 1000,
                        const_dz=True,
                        print_current_state_every=1,
                        max_intensity_to_stop=5 * 10**17,
                        plot_beam_every=0,
                        plot_spectrum_every=5,
                        spectrum_visualizer=spectrum_visualizer,
                        save_field=True)

# initiate propagation process
propagator.propagate()
