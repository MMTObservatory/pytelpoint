# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord, Angle, AltAz
from astropy.time import Time

from tpoint.constants import MMT_LOCATION


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
        has the header info that the old TPOINT program requires. This should be 0 for a bare file
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
    aa_frame = AltAz(obstime=obstime, location=MMT_LOCATION)
    t = ascii.read(filename, data_start=data_start, format='no_header', guess=False, fast_reader=False)
    az_obs = Angle(t['col1'], unit=u.degree).wrap_at(360 * u.deg)
    el_obs = Angle(t['col2'], unit=u.degree).wrap_at(360 * u.deg)
    az_raw = Angle(t['col3'], unit=u.degree).wrap_at(360 * u.deg)
    el_raw = Angle(t['col4'], unit=u.degree).wrap_at(360 * u.deg)
    coo_obs = SkyCoord(az_obs, el_obs, frame=aa_frame)
    coo_raw = SkyCoord(az_raw, el_raw, frame=aa_frame)

    return coo_obs, coo_raw
