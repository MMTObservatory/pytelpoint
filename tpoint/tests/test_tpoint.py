# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import assert_quantity_allclose as assert_allclose
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy.time import Time
import astropy.units as u

from ..transform import tpoint
from ..constants import MMT_LOCATION


def test_tpoint():
    obstime = Time("2021-08-21T06:00:00", format='isot')
    aa_frame = AltAz(obstime=obstime, location=MMT_LOCATION)
    coo_raw = SkyCoord(180 * u.deg, 45 * u.deg, frame=aa_frame)
    coo_trans = tpoint(coo_raw, ia=1.0)
    assert_allclose(coo_trans.az - coo_raw.az, Angle(-1. * u.arcsec))
