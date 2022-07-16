Documentation
=============

Pytelpoint uses `PyMC <https://www.pymc.io/>`_ to perform robust analysis of telescope pointing performance.
It implements pointing models in a way similar to `TPOINT <https://www.bisque.com/product/tpoint/>`_ and uses compatible
parameter names and definitions. This way results can be easily compared with previous TPOINT analysis and implemented
in telescope control systems that use TPOINT or TPOINT-compatible pointing models. The Bayesian modeling techniques that
PyMC enables allow for much more robust determinations of the uncertainties in model parameters and the correlations
between them. Several visualization routines are provided to help assess results and the residuals of the model fits.
The initial release only supports elevation-azimuth telescopes. Support for other kinds of mounts is planned.


API Documentation
-----------------
.. toctree::
  :maxdepth: 2

  pytelpoint/api.rst

.. note:: pytelpoint is in the early stages of development. API changes are to be expected
          and new features/functionality will be added frequently.
