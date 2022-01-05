# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
from astropy.coordinates import EarthLocation

__all__ = ['MMT_LOCATION']


MMT_LOCATION = EarthLocation.from_geodetic("-110:53:04.4", "31:41:19.6", 2600 * u.m)
