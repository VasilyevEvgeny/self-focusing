from tqdm import tqdm
from argparse import Namespace
from os import mkdir

from core import BeamR, Propagator, SweepDiffractionExecutorR, KerrExecutorR, create_multidir, xlsx_to_df
from tests.vortex_critical_power.test_vortex_critical_power import TestVortexCriticalPower

NAME = 'vortex_critical_power_r'


class TestVortexCriticalPowerR(TestVortexCriticalPower):
    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)

        self.add_prefix(NAME)
        self.__results_dir, self.__results_dir_name = create_multidir(self._args.global_root_dir,
                                                                      self._args.global_results_dir_name,
                                                                      self._args.prefix)

        self._n_z = 5000

        self._eps = 0.15
        self._png_name = NAME

    def process(self):
        for idx, m in enumerate(tqdm(self._ms, desc=NAME)):
            m_dir_name = 'm=%d' % m
            m_dir = self.__results_dir + '/' + m_dir_name
            mkdir(m_dir)

            dfs = []
            p_v_pred = None
            flag_critical_found = False

            for p_v_normalized in self._p_vs:

                beam = BeamR(medium=self._medium.info,
                             M=m,
                             m=m,
                             p_0_to_p_vortex=p_v_normalized,
                             lmbda=self._lmbda,
                             r_0=self._radius,
                             n_r=2048)

                global_results_dir_name = m_dir.split(self._args.global_root_dir)[1][1:]
                args = Namespace(global_root_dir=self._args.global_root_dir,
                                 global_results_dir_name=global_results_dir_name,
                                 prefix='p_v_to_p_v_true=%2.2f' % p_v_normalized,
                                 insert_datetime=False)

                i_max_to_stop = self._n_i_max_to_stop * beam.i_0 * beam.i_max
                propagator = Propagator(args=args,
                                        beam=beam,
                                        diffraction=SweepDiffractionExecutorR(beam=beam),
                                        kerr_effect=KerrExecutorR(beam=beam),
                                        n_z=self._n_z,
                                        dz0=self._n_z_diff * beam.z_diff / self._n_z,
                                        flag_const_dz=True,
                                        dn_print_current_state=0,
                                        dn_plot_beam=0,
                                        beam_normalization_type='local',
                                        max_intensity_to_stop=i_max_to_stop)

                propagator.propagate()

                if not flag_critical_found and propagator.beam.i_max * propagator.beam.i_0 > i_max_to_stop:
                    p_v_pred = p_v_normalized
                    flag_critical_found = True

                df = xlsx_to_df(propagator.logger.track_filename,
                                normalize_z_to=1,
                                normalize_i_to=i_max_to_stop / self._n_i_max_to_stop)
                df['z_normalized'] /= propagator.beam.z_diff
                dfs.append((p_v_normalized, df))

                del beam
                del propagator

            self._p_v_rel_pred[idx] = p_v_pred

            self.plot_propagation(dfs, m_dir, m)

    def test_diffraction_r_gauss(self):
        self.process()
        self.check()

        if self._flag_plot:
            self.plot(self.__results_dir)
