# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources

from tpoint.io import read_azel_datfile


def test_read_azel_datfile():
    test_file = pkg_resources.resource_filename("tpoint", os.path.join("test_data", "k_and_e.dat"))
    coords = read_azel_datfile(test_file)
    assert(coords is not None)

    bogus = read_azel_datfile("bazz")
    assert(bogus is None)
