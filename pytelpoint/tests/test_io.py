# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources

from astropy.tests.helper import assert_quantity_allclose as assert_allclose

from pytelpoint.io import read_azel_datfile, read_raw_datfile


def test_read_azel_datfile():
    proc_file = pkg_resources.resource_filename("pytelpoint", os.path.join("test_data", "k_and_e.dat"))
    raw_file = pkg_resources.resource_filename("pytelpoint", os.path.join("test_data", "point_20210821.dat"))

    proc_coords = read_azel_datfile(proc_file)
    assert(proc_coords is not None)

    bogus = read_azel_datfile("bazz")
    assert(bogus is None)

    raw_coords = read_raw_datfile(raw_file)
    assert(raw_coords is not None)

    raw_bogus = read_raw_datfile("bazz")
    assert(raw_bogus is None)

    # k_and_e.dat is processed from point_20210821.dat using the old ruby script. the observed azimuths
    # had better match.
    assert_allclose(proc_coords[0].az, raw_coords[0].az)
