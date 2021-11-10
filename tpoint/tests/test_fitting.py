# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources

from tpoint.fitting import mc_tpoint, best_fit_pars
from tpoint.visualization import plot_corner, plot_posterior
from tpoint.io import read_azel_datfile


def test_mc_fitting():
    test_file = pkg_resources.resource_filename("tpoint", os.path.join("test_data", "k_and_e.dat"))
    coo_ref, coo_meas = read_azel_datfile(test_file)
    idata = mc_tpoint(coo_ref, coo_meas, nsamp=200, ntune=200)
    assert(idata is not None)

    pfig = plot_posterior(idata)
    assert(pfig is not None)

    cfig = plot_corner(idata)
    assert(cfig is not None)

    tpars = best_fit_pars(idata)
    assert(tpars is not None)
