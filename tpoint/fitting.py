# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import pymc3 as pm


def mc_tpoint(coo_ref, coo_meas, nsamp=20000, ntune=4000, target_accept=0.95, random_seed=8675309):
    """
    Fit full tpoint az/el model using PyMC3. This fit includes the eight normal terms used in
    `~tpoint.transform.tpoint` with an additional term, sigma, that describes the intrinsic scatter.

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_meas : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    nsamp : int (default: 20000)
        Number of inference samples
    ntune : int (default: 4000)
        Number of burn-in samples
    target_accept : float (default: 0.95)
        Sets acceptance probability target for determining step size
    random_seed : int (default: 8675309)
        Seed number for random number generator

    Returns
    -------
    idata : `~arviz.InferenceData`
        Inference data from the tpoint model
    """
    tpoint_model = pm.Model()
    deg2rad = np.pi / 180
    with tpoint_model:
        az = pm.Data('az', coo_ref.az)
        el = pm.Data('el', coo_ref.alt)
        az_raw = pm.Data('az_raw', coo_meas.az)
        el_raw = pm.Data('el_raw', coo_meas.alt)

        ia = pm.Normal('ia', 1200., 100)
        ie = pm.Normal('ie', 0., 50.)
        an = pm.Normal('an', 0., 20.)
        aw = pm.Normal('aw', 0., 20.)
        ca = pm.Normal('ca', 0., 30.)
        npae = pm.Normal('npae', 0., 30.)
        tf = pm.Normal('tf', 0., 50.)
        tx = pm.Normal('tx', 0., 20.)
        sigma = pm.HalfNormal('sigma', 1.)

        daz = -ia
        daz -= an * pm.math.sin(deg2rad * az) * pm.math.tan(deg2rad * el)
        daz -= aw * pm.math.cos(deg2rad * az) * pm.math.tan(deg2rad * el)
        daz -= ca / pm.math.cos(deg2rad * el)
        daz -= npae * pm.math.tan(deg2rad * el)

        dalt = ie
        dalt -= an * pm.math.cos(deg2rad * az)
        dalt += aw * pm.math.sin(deg2rad * az)
        dalt -= tf * pm.math.cos(deg2rad * el)
        dalt -= tx / pm.math.tan(deg2rad * el)

        separations = ((az - (az_raw + daz/3600.)) ** 2 + (el - (el_raw + dalt/3600.)) ** 2) ** 0.5

        _ = pm.HalfNormal('likelihood', sigma=sigma/3600, observed=separations)

        idata = pm.sample(
            nsamp,
            tune=ntune,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=random_seed
        )
    return idata
