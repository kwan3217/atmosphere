"""
Test the SimpleEarthAtmosphere model

Created: 1/23/25
"""
import numpy as np
import pytest

from atmosphere.atmosphere import SimpleEarthAtmosphere


@pytest.mark.parametrize(
    "alt",
    [0.0,12000.0,149999.0,151000.0,np.array((0,151000))]
)
def test_simple_atmosphere(alt):
    earth_atm=SimpleEarthAtmosphere()
    print(earth_atm.calc_props(alt))