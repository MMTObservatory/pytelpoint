# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import arviz
import pymc as pm

import astropy.units as u

__all__ = ['azel_fit', 'best_fit_pars']

DEG2RAD = np.pi / 180
AZEL_TERMS = ('ia', 'ie', 'an', 'aw', 'ca', 'npae', 'tf', 'tx')
HADEC_TERMS = ('ih', 'id', 'np', 'ch', 'ma', 'me', 'tf')

AZIMUTH_FUNCS = {
    'ia': lambda az, el: -1.0,
    'an': lambda az, el: -1.0 * pm.math.sin(DEG2RAD * az) * pm.math.tan(DEG2RAD * el),
    'aw': lambda az, el: -1.0 * pm.math.cos(DEG2RAD * az) * pm.math.tan(DEG2RAD * el),
    'ca': lambda az, el: -1.0 / pm.math.cos(DEG2RAD * el),
    'npae': lambda az, el: -1.0 * pm.math.tan(DEG2RAD * el)
}

ELEVATION_FUNCS = {
    'ie': lambda az, el: 1.0,
    'an': lambda az, el: -1.0 * pm.math.cos(DEG2RAD * az),
    'aw': lambda az, el: pm.math.sin(DEG2RAD * az),
    'tf': lambda az, el: -1.0 * pm.math.cos(DEG2RAD * el),
    'tx': lambda az, el: -1.0 / pm.math.tan(DEG2RAD * el)
}


def azel_fit(
        coo_ref,
        coo_meas,
        nsamp=2000,
        ntune=500,
        target_accept=0.95,
        random_seed=8675309,
        cores=None,
        fit_terms=AZEL_TERMS,
        fixed_terms=None,
        init_pars=None,
        prior_sigmas=None
):
    """
    Fit full az/el pointing model using PyMC. The terms are analogous to those used by TPOINT(tm). This fit includes
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
    fit_terms : list-like (default: AZEL_TERMS)
        Model terms to include in the fit.
    fixed_terms : dict (default: {})
        Dict of terms to fix to a specified value.
    init_pars : dict (default: None -> {'ia': 1200.})
        Initial guesses for the fit parameters. Keys are the same those provided by
        `~pytelpoint.fitting.best_fit_pars` and described in `~pytelpoint.transform.azel`:
        'ia', 'ie', 'an', 'aw', 'ca', 'npae', 'tf', 'tx', 'az_sigma', 'el_sigma'.
        The default for 'ia' is appropriate for the MMTO. If not specified, then the initial
        guess for a parameter is assumed to be 0.
    prior_sigmas : dict (default: None -> {'ia': 100., 'ie': 50.})
        The priors for the fit parameters are assumed to be `~pymc.Normal` distributions. The sigmas
        for these can be specified here. The index parameters, 'ia' and 'ie', have default sigma values
        of 100 and 50, respectively. The rest default to 25 if not specified.

    Returns
    -------
    idata : `~arviz.InferenceData`
        Inference data from the pointing model
    """
    if fixed_terms is None:
        fixed_terms = {}
    if init_pars is None:
        init_pars = {'ia': 1200.}
    if prior_sigmas is None:
        prior_sigmas = {'ia': 100., 'ie': 50.}

    pointing_model = pm.Model()

    with pointing_model:
        # az/el are the astrometric reference values. az_raw/el_raw are the observed encoder values.
        # they should be in degrees, but are converted here just in case.
        az = pm.ConstantData('az', coo_ref.az.to(u.deg).value)
        el = pm.ConstantData('el', coo_ref.alt.to(u.deg).value)
        az_raw = pm.ConstantData('az_raw', coo_meas.az.to(u.deg).value)
        el_raw = pm.ConstantData('el_raw', coo_meas.alt.to(u.deg).value)

        terms = {}

        combined_terms = list(fit_terms) + list(fixed_terms.keys())
        for term in combined_terms:
            if term not in AZEL_TERMS:
                raise ValueError(f"Invalid az/el fitting term, {term}.")
            if term in fixed_terms:
                terms[term] = fixed_terms[term]
            else:
                terms[term] = pm.Normal(term, init_pars.get(term, 0.0), prior_sigmas.get(term, 25.))

        az_sigma = pm.HalfNormal('az_sigma', sigma=init_pars.get('az_sigma', 1.))
        el_sigma = pm.HalfNormal('el_sigma', sigma=init_pars.get('el_sigma', 1.))

        daz = 0.0
        for k, f in AZIMUTH_FUNCS.items():
            if k in combined_terms:
                daz += terms[k] * f(az, el)

        # using 'dalt' because 'del' is a python built-in
        dalt = 0.0
        for k, f in ELEVATION_FUNCS.items():
            if k in combined_terms:
                dalt += terms[k] * f(az, el)

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
