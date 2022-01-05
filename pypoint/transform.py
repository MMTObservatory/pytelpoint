# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

__all__ = ['azel_model']


def azel_model(
        coo,
        ia=0.,
        ie=0.,
        an=0.,
        aw=0.,
        ca=0.,
        npae=0.,
        tf=0.,
        tx=0.,
        **kwargs):
    """
    Apply 8-term alt-az pointing model to set of raw alt-az coordinates and return corrected coordinates.
    Parameter names match those used by TPOINT(tm): IA, IE, AN, AW, CA, NPAE, TF, and TX. See TPOINT(tm) documentation
    for more details.

    Parameters
    ----------
    coo : `~astropy.coordinates.SkyCoord` instance
        Raw Az-El coordinates to correct via pointing model model. Must be in an AltAz frame.
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

    Returns
    -------
    new_coo : `~astropy.coordinates.SkyCoord` instance
        New coordinates with azel pointing model applied.
    """
    if coo.frame.name != 'altaz':
        raise ValueError("azel_model can only be applied to AltAz coordinates.")

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

    new_coo = SkyCoord(new_az, new_alt, frame=coo.frame)
    return new_coo
