# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time


MMT_LOCATION = EarthLocation.from_geodetic("-110:53:04.4", "31:41:19.6", 2600 * u.m)
#obstime = Time("2021-08-21T06:00:00", format='isot')

def tpoint(
        coo,
        obstime=Time.now(),
        ia=0.,
        ie=0.,
        an=0.,
        aw=0.,
        ca=0.,
        npae=0.,
        tf=0.,
        tx=0.
    ):
    """
    Apply 8-term alt-az TPOINT model to set of raw alt-az coordinates and return corrected coordinates.
    Parameter names match those used by TPOINT: IA, IE, AN, AW, CA, NPAE, TF, and TX. See TPOINT documentation
    for more details.

    Parameters
    ----------
    coo : `~astropy.coordinates.SkyCoord` instance
        Raw Az-El coordinates to correct via TPOINT model
    obstime : `~astropy.time.Time` instance (default: `~astropy.time.Time.now()`)
        Observation time of the raw coordinates
    ia : float (default: 0)
        Azimuth index value (i.e. zeropoint)
    ie : float (default: 0)
        Elevation index value
    an : float (default: 0)
        North-South misalignment of azimuth axis
    aw : float (default: 0)
        East-West misalignment of azimuth axis
    ca : float (default: 0)
        Left-Right collimation error. In an alt-az mount, this collimation error is the non-perpendicularity
        between the nominated pointing direction and the elevation axis. It produces a left-right shift on the
        sky that is constant for all elevations.
    npae : float (default: 0)
        Az/El non-perpendicularity. In an alt-az mount, if the azimuth and elevation axes are not exactly at
        right angles, horizontal shifts occur that are proportional to sin(el).
    tf : float (default: 0)
        Tube flexure term proportional to cos(el).
    tx : float (default: 0)
        Tube flexure term proportional to cot(el).
    """
    # don't strictly need time or location for our purposes, but astropy wants them for defining the alt/az frame
    aa_frame = AltAz(obstime=obstime, location=MMT_LOCATION)

    da = -1 * ia
    da -= an * np.sin(coo.az) * np.tan(coo.alt)
    da -= aw * np.cos(coo.az) * np.tan(coo.alt)
    da -= ca / np.cos(coo.alt)
    da -= npae * np.tan(coo.alt)

    de = ie
    de -= an * np.cos(coo.az)
    de += aw * np.sin(coo.az)
    de -= tf * np.cos(coo.alt)
    de -= tx / np.tan(coo.alt)

    new_az = coo.az + da * u.arcsec
    new_alt = coo.alt + de * u.arcsec

    new_coo = SkyCoord(new_az, new_alt, frame=aa_frame)
    return new_coo