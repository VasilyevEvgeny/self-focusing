from core import BeamR, SpectrumR, SweepDiffractionExecutorR, KerrExecutorR, Propagator, SpectrumVisualizer, \
    parse_args, make_animation, make_video

# parse args from command line
args = parse_args()

# create object of 3D axisymmetric beam
beam = BeamR(medium='LiF',
             p_0_to_p_vortex=5,
             m=1,
             M=1,
             lmbda=1800*10**-9,
             r_0=100*10**-6,
             radii_in_grid=70,
             n_r=4096)

spectrum = SpectrumR(beam=beam)

# create visualizer object
spectrum_visualizer = SpectrumVisualizer(spectrum=spectrum,
                                         log_scale_of_spectrum=False,
                                         remaining_central_part_coeff_field=0.04,     # 0.05
                                         remaining_central_part_coeff_spectrum=0.03)  # 0.12

# create propagator object
propagator = Propagator(args=args,
                        beam=beam,
                        spectrum=spectrum,
                        diffraction=SweepDiffractionExecutorR(beam=beam),
                        kerr_effect=KerrExecutorR(beam=beam),
                        n_z=1001,
                        dz_0=beam.z_diff / 1000,
                        const_dz=False,
                        print_current_state_every=1,
                        max_intensity_to_stop=5 * 10**17,
                        plot_beam_every=0,
                        plot_spectrum_every=5,
                        spectrum_visualizer=spectrum_visualizer,
                        save_field=False,
                        save_spectrum=False)

# initiate propagation process
propagator.propagate()

# multimedia
make_animation(root_dir=propagator.manager.results_dir,
               images_dir_name='beam',
               name=args.prefix,
               fps=10)

make_video(root_dir=propagator.manager.results_dir,
           images_dir_name='beam',
           name=args.prefix,
           fps=10)
