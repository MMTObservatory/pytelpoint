# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import astropy.units as u

__all__ = ["skyrms", "psd"]


def skyrms(coo_ref, coo_meas):
    """
    Calculate sky RMS of the offsets between reference and measured coordinates in the same way as TPOINT(tm).
    Return the result in arcseconds.

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_meas : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates

    Returns:
    --------
    rms : `~astropy.units.Quantity` (u.arcsec)
        Root mean squared of the separation between coo_ref and coo_meas expressed in arcseconds.
    """
    sep = coo_ref.separation(coo_meas)
    rms = np.sqrt((sep ** 2).mean()).to(u.arcsec)
    return rms


def psd(coo_ref, coo_meas, nterms=8):
    """
    Calculate the population standard deviation, PSD, the way TPOINT(tm) does. Return the result in arcseconds.

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_meas : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    nterms : int (default: 8)
        Number of terms used in the model used to correct coo_meas to match coo_ref

    Returns:
    --------
    sd : `~astropy.units.Quantity` (u.arcsec)
        Population SD of the separation between coo_ref and coo_meas expressed in arcseconds.
    """
    rms = skyrms(coo_ref, coo_meas)
    sd = np.sqrt(rms**2 * len(coo_meas) / (len(coo_meas) - nterms))
    return sd
