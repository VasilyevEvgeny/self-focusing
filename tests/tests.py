from tests.diffraction.test_diffraction_x_gauss import TestDiffractionXGauss
from tests.diffraction.test_diffraction_r_gauss import TestDiffractionRGauss
from tests.diffraction.test_diffraction_xy_gauss import TestDiffractionXYGauss
from tests.diffraction.test_diffraction_r_vortex import TestDiffractionRVortex
from tests.diffraction.test_diffraction_xy_vortex import TestDiffractionXYVortex


class TestAll:
    def test_all(self):
        TestDiffractionXGauss()
        TestDiffractionRGauss()
        TestDiffractionXYGauss()
        TestDiffractionRVortex()
        TestDiffractionXYVortex()
