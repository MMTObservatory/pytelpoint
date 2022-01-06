# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources

from pytelpoint.fitting import azel_fit, best_fit_pars
from pytelpoint.visualization import plot_corner, plot_posterior
from pytelpoint.io import read_azel_datfile


def test_mc_fitting():
    test_file = pkg_resources.resource_filename("pytelpoint", os.path.join("test_data", "k_and_e.dat"))
    coo_ref, coo_meas = read_azel_datfile(test_file)
    idata = azel_fit(coo_ref, coo_meas, nsamp=200, ntune=200)
    assert(idata is not None)

    pfig = plot_posterior(idata)
    assert(pfig is not None)

    cfig = plot_corner(idata)
    assert(cfig is not None)

    tpars = best_fit_pars(idata)
    assert(tpars is not None)
