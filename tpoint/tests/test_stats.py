# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy
import numpy as np

from astropy.tests.helper import assert_quantity_allclose as assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, AltAz, SkyCoord
from astropy.time import Time

from tpoint.stats import skyrms, psd
from tpoint.constants import MMT_LOCATION
from tpoint.transform import tpoint


OBSTIME = Time("2021-08-21T06:00:00", format='isot')
AA_FRAME = AltAz(obstime=OBSTIME, location=MMT_LOCATION)
AZ_REF = Angle([0, 45, 90, 135, 180, 225, 270, 315, 360], unit=u.degree).wrap_at(360 * u.deg)
EL_REF = Angle([15, 20, 30, 45, 50, 60, 65, 75, 85], unit=u.degree).wrap_at(360 * u.deg)
COO_REF = SkyCoord(AZ_REF, EL_REF, frame=AA_FRAME)
COO_MOD = tpoint(COO_REF, ia=1.)


def test_skyrms():
    rms = skyrms(COO_REF, COO_MOD)
    assert_allclose(rms, 0.665198 * u.arcsec, rtol=1e-5)


def test_psd():
    p = psd(COO_REF, COO_MOD, nterms=8)
    assert_allclose(p, 1.995595 * u.arcsec, rtol=1e-5)