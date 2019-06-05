from tqdm import tqdm

from core import BeamR, Propagator, SweepDiffractionExecutorR, KerrExecutorR, create_multidir
from tests.marburger.test_marburger import TestMarburger

NAME = 'marburger_r'


class TestMarburgerR(TestMarburger):
    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)

        self._add_prefix(NAME)
        self.__results_dir, self.__results_dir_name = create_multidir(self._args.global_root_dir, self._args.global_results_dir_name,
                                                                      self._args.prefix)

        self._n_z = 10000

        self._eps = 0.3
        self._png_name = NAME

    def process(self):
        for idx, p_rel in enumerate(tqdm(self._p_rels_for_pred, desc=NAME)):
            beam = BeamR(medium=self._medium.info,
                         M=0,
                         m=0,
                         p_0_to_p_gauss=p_rel,
                         lmbda=self._lmbda,
                         r_0=self._radius,
                         n_r=2048)

            propagator = Propagator(args=self._args,
                                    multidir_name=self.__results_dir_name,
                                    beam=beam,
                                    diffraction=SweepDiffractionExecutorR(beam=beam),
                                    kerr_effect=KerrExecutorR(beam=beam),
                                    n_z=self._n_z,
                                    dz0=beam.z_diff / self._n_z,
                                    flag_const_dz=False,
                                    dn_print_current_state=0,
                                    dn_plot_beam=0,
                                    beam_normalization_type='local',
                                    max_intensity_to_stop=10**17)

            propagator.propagate()

            self._z_fil_rel_pred[idx] = propagator.z / propagator.beam.z_diff

            del beam
            del propagator

    def test_marburger_r(self):
        self.process()
        self._check()

        if self._flag_plot:
            self._plot(self.__results_dir)
