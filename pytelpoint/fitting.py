# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import arviz
import pymc as pm

__all__ = ['azel_fit', 'best_fit_pars']


def azel_fit(
        coo_ref,
        coo_meas,
        nsamp=2000,
        ntune=500,
        target_accept=0.95,
        random_seed=8675309,
        cores=None,
        init_pars={}):
    """
    Fit full az/el pointing model using PyMC3. The terms are analogous to those used by TPOINT(tm). This fit includes
    the eight normal terms used and described in `~pytelpoint.transform.azel` with additional terms,
    az_sigma and el_sigma, that describeq the intrinsic/observational scatter.

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_meas : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    nsamp : int (default: 2000)
        Number of inference samples per chain
    ntune : int (default: 500)
        Number of burn-in samples per chain
    target_accept : float (default: 0.95)
        Sets acceptance probability target for determining step size
    random_seed : int (default: 8675309)
        Seed number for random number generator
    cores : int (default: None)
        Number of cores to use for parallel chains. The default of None
        will use the number of available cores, but no more than 4.
    init_pars : dict (default: {})
        Initial guesses for the fit parameters. Keys are the same those provided by
        `~pytelpoint.fitting.best_fit_pars` and described in `~pytelpoint.transform.azel`:
        'ia', 'ie', 'an', 'aw', 'ca', 'npae', 'tf', 'tx', 'az_sigma', 'el_sigma'

    Returns
    -------
    idata : `~arviz.InferenceData`
        Inference data from the pointing model
    """
    pointing_model = pm.Model()
    deg2rad = np.pi / 180
    with pointing_model:
        # az/el are the astrometric reference values. az_raw/el_raw are the observed encoder values.
        az = pm.ConstantData('az', coo_ref.az.value)
        el = pm.ConstantData('el', coo_ref.alt.value)
        az_raw = pm.ConstantData('az_raw', coo_meas.az.value)
        el_raw = pm.ConstantData('el_raw', coo_meas.alt.value)

        ia = pm.Normal('ia', init_pars.get('ia', 1200.), 100)
        ie = pm.Normal('ie', init_pars.get('ie', 0.), 50.)
        an = pm.Normal('an', init_pars.get('an', 0.), 20.)
        aw = pm.Normal('aw', init_pars.get('aw', 0.), 20.)
        ca = pm.Normal('ca', init_pars.get('ca', 0.), 30.)
        npae = pm.Normal('npae', init_pars.get('npae', 0.), 30.)
        tf = pm.Normal('tf', init_pars.get('tf', 0.), 50.)
        tx = pm.Normal('tx', init_pars.get('tx', 0.), 20.)
        az_sigma = pm.HalfNormal('az_sigma', sigma=init_pars.get('az_sigma', 1.))
        el_sigma = pm.HalfNormal('el_sigma', sigma=init_pars.get('el_sigma', 1.))

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

        # models are the raw encoder values plus pointing model; observed are the actual az/el
        mu_az = az_raw + daz/3600.
        mu_el = el_raw + dalt/3600.
        _ = pm.Normal('azerr', mu=mu_az, sigma=az_sigma/3600, observed=az)
        _ = pm.Normal('elerr', mu=mu_el, sigma=el_sigma/3600, observed=el)

        idata = pm.sample(
            nsamp,
            tune=ntune,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=random_seed,
            cores=cores
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
    for p in t_fit.index:
        pointing_pars[p] = t_fit.loc[p, 'mean']

    return pointing_pars
