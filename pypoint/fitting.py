# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import arviz
import pymc3 as pm


def azel_point(coo_ref, coo_meas, nsamp=2000, ntune=2000, target_accept=0.95, random_seed=8675309):
    """
    Fit full az/el pointing model using PyMC3. The terms are analogous to those used by TPOINT(tm). This fit includes
    the eight normal terms used in `~pypoint.transform.azel` with additional terms, az_sigma and el_sigma, that
    describes the intrinsic scatter.

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_meas : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    nsamp : int (default: 2000)
        Number of inference samples
    ntune : int (default: 2000)
        Number of burn-in samples
    target_accept : float (default: 0.95)
        Sets acceptance probability target for determining step size
    random_seed : int (default: 8675309)
        Seed number for random number generator

    Returns
    -------
    idata : `~arviz.InferenceData`
        Inference data from the pointing model
    """
    pointing_model = pm.Model()
    deg2rad = np.pi / 180
    with pointing_model:
        # az/el are the astrometric reference values. az_raw/el_raw are the observed encoder values.
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
        az_sigma = pm.HalfNormal('az_sigma', sigma=1.)
        el_sigma = pm.HalfNormal('el_sigma', sigma=1.)

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

        _ = pm.Normal('azerr', mu=0., sigma=az_sigma/3600, observed=pm.math.cos(deg2rad * el) * (az - (az_raw + daz/3600.)))
        _ = pm.Normal('elerr', mu=0., sigma=el_sigma/3600, observed=el - (el_raw + dalt/3600.))

        idata = pm.sample(
            nsamp,
            tune=ntune,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=random_seed
        )
    return idata


def best_fit_pars(idata):
    """
    Pull out the best fit parameters from a pointing model fit and return them as a dict.

    Parameters
    ----------
    idata : `~arviz.InferenceData`
        Inference data from pointing model.

    Returns
    -------
    pointing_pars : dict
        Best-fit pointing parameters
    """
    t_fit = arviz.summary(idata, round_to=8)
    pointing_pars = {}
    for p in ['ia', 'ie', 'an', 'aw', 'ca', 'npae', 'tf', 'tx', 'az_sigma', 'el_sigma']:
        pointing_pars[p] = t_fit.loc[p, 'mean']

    return pointing_pars
