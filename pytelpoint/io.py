# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord, Angle, AltAz
from astropy.time import Time

from pytelpoint.constants import MMT_LOCATION

__all__ = ['read_azel_datfile', 'read_raw_datfile']


def _mk_azel_coords(az_ref, el_ref, az_meas, el_meas, obstime=Time.now()):
    """
    Take reference and measaured arrays of az and el, assumed to be in degrees, construct Angles from them,
    and return reference and measured SkyCoord objects.

    Parameters
    ----------
    az_ref, el_ref, az_meas, el_meas : array-like or list-like
        Arrays containing reference azimuth, reference elevation, measured azimuth, and measured elevation.
    obstime : `~astropy.time.Time`
        Time the data were taken. This is pretty strictly optional since mount code takes this into
        account when calculating the astrometric az/el for each target.

    Returns
    -------
    coo_ref, coo_meas : `~astropy.coordinates.SkyCoord` instances
        Actual az/el coordinates for each target, measured az/el coordinates as reported by, e.g.,
        az/el encoders
    """
    aa_frame = AltAz(obstime=obstime, location=MMT_LOCATION)
    az_ref = Angle(az_ref, unit=u.degree).wrap_at(360 * u.deg)
    el_ref = Angle(el_ref, unit=u.degree).wrap_at(360 * u.deg)
    az_meas = Angle(az_meas, unit=u.degree).wrap_at(360 * u.deg)
    el_meas = Angle(el_meas, unit=u.degree).wrap_at(360 * u.deg)
    coo_ref = SkyCoord(az_ref, el_ref, frame=aa_frame)
    coo_meas = SkyCoord(az_meas, el_meas, frame=aa_frame)

    return coo_ref, coo_meas


def read_azel_datfile(filename, data_start=20, obstime=Time.now()):
    """
    This reads in a processed MMTO pointing run data file that has four columns:

    az observed, alt observed, az raw, el raw

    where the observed values are calculated from the targets' astronomy and observation time and the
    raw values are as reported by the axis encoders. All values in units of degrees.

    Parameters
    ----------
    filename : str
        Name of the data file to read
    data_start : int (default: 20)
        Where the pointing data starts within the file. This should be the default of 20 if the file
        has the header info that the old TPOINT(tm) program requires. This should be 0 for a bare file
        that doesn't contain that.
    obstime : `~astropy.time.Time`
        Time the data were taken. This is pretty strictly optional since mount code takes this into
        account when calculating the astrometric az/el for each target.

    Returns
    -------
    coo_obs, coo_raw : `~astropy.coordinates.SkyCoord` instances
        Actual observed az/el coordinates for each target, raw az/el coordinates as reported by
        the az/el encoders
    """
    try:
        t = ascii.read(filename, data_start=data_start, format='no_header', guess=False, fast_reader=False)

        coo_obs, coo_raw = _mk_azel_coords(t['col1'], t['col2'], t['col3'], t['col4'], obstime=obstime)

        return coo_obs, coo_raw
    except Exception as e:
        print(f"Problem reading in {filename}: {e}")
        return None


def read_raw_datfile(filename, obstime=Time.now()):
    """
    This reads the raw pointing data files as produced by the MMTO mount code.

    Parameters
    ----------
    filename : str
        Name of the data file to read
    obstime : `~astropy.time.Time`
        Time the data were taken. This is pretty strictly optional since mount code takes this into
        account when calculating the astrometric az/el for each target.

    Returns
    -------
    coo_obs, coo_raw : `~astropy.coordinates.SkyCoord` instances
        Actual observed az/el coordinates for each target, raw az/el coordinates as reported by
        the az/el encoders
    """
    try:
        az_obs, el_obs, az_raw, el_raw = [], [], [], []

        with open(filename, 'r') as fp:
            lines = fp.readlines()

        # flag for raw encoder position
        a_re = re.compile("^RAA:")

        # flag for calculated position
        t_re = re.compile("^TAA:")

        for line in lines:
            if a_re.match(line):
                line_data = line.split()
                az_raw.append(180.0 - float(line_data[2]))
                el_raw.append(float(line_data[1]))
            if t_re.match(line):
                line_data = line.split()
                az_obs.append(180.0 - float(line_data[2]))
                el_obs.append(line_data[1])

        coo_obs, coo_raw = _mk_azel_coords(az_obs, el_obs, az_raw, el_raw, obstime=obstime)

        return coo_obs, coo_raw
    except Exception as e:
        print(f"Problem reading in raw pointing data {filename}: {e}")
        return None
