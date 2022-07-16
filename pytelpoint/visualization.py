# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import corner
import arviz

import astropy.units as u
from astropy.visualization import hist

from pytelpoint.stats import psd, skyrms
from pytelpoint.fitting import best_fit_pars

__all__ = [
    'pointing_histogram',
    'pointing_residuals',
    'pointing_sky',
    'pointing_azel_resid',
    'plot_corner',
    'plot_posterior'
]


def pointing_histogram(coo_ref, coo_meas, bins='freedman', style='ggplot'):
    """
    Plot histogram of separations between reference coordinates, coo_ref, and measured coordinates, coo_meas.

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_meas : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    bins : str (default: freedman)
        Name of algorithm to use to calculate optimal bin sizes. See `~astropy.visualization.hist` for options
        and descriptions.
    style : str (default: ggplot)
        Matplotlib style to apply to the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance
        Figure object containing the histogram plot.
    """
    seps = coo_ref.separation(coo_meas)
    with plt.style.context(style, {'xtick.labelsize': 18, 'ytick.labelsize': 18}):
        matplotlib.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(figsize=[9, 6])
        hist(seps.to(u.arcsec).value, bins=bins, ax=ax, histtype='stepfilled', alpha=0.6)
        ax.set_ylabel("N")
        ax.set_xlabel("Pointing Error (arcsec)")
        med = np.median(seps.to(u.arcsec))
        rms = skyrms(coo_ref, coo_meas)
        skypsd = psd(coo_ref, coo_meas)
        ax.vlines(
            x=med.value,
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            color='gray',
            alpha=0.5,
            ls='-',
            label=f"Median: {med.value:.2f}\""
        )
        ax.vlines(
            x=rms.value,
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            color='gray',
            alpha=0.5,
            ls='--',
            label=f"RMS: {rms.value:.2f}\""
        )
        ax.vlines(
            x=skypsd.value,
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            color='gray',
            alpha=0.5,
            ls=':',
            label=f"PSD: {skypsd.value:.2f}\""
        )
        ax.legend()
        plt.tight_layout()
    return fig


def pointing_residuals(coo_ref, coo_meas, circle_size=1.0, style="ggplot"):
    """
    Plot of pointing residuals in az/el space

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_mes : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    circle_size : float (default: 1.)
        Size of reference circle to plot with residuals. Set to None
        for no circle.
    style : str (default: ggplot)
        Matplotlib style to apply to the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance
        Figure object containing the residuals plot.
    """
    az_res, el_res = coo_meas.spherical_offsets_to(coo_ref)
    with plt.style.context(style, {'xtick.labelsize': 18, 'ytick.labelsize': 18}):
        matplotlib.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.set_aspect('equal')
        ax.scatter(az_res.to(u.arcsec), el_res.to(u.arcsec))
        if circle_size is not None:
            ax_size = 5 * circle_size
            ax.set_xlim([-ax_size, ax_size])
            ax.set_ylim([-ax_size, ax_size])
        ax.set_xlabel(r"$\Delta$A (arcsec)")
        ax.set_ylabel(r"$\Delta$E (arcsec)")
        ax.set_title("Azimuth-Elevation Residuals")
        if circle_size is not None:
            c1 = matplotlib.patches.Circle(
                    (0, 0),
                    circle_size,
                    ec='black',
                    lw=4,
                    fill=False,
                    alpha=0.4,
                    label=f"{circle_size}\""
                )
            ax.add_patch(c1)
            ax.legend()
        plt.tight_layout()
    return fig


def pointing_sky(coo_ref, coo_meas, style="ggplot"):
    """
    Plot of pointing errors as a function of sky position

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_mes : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    style : str (default: ggplot)
        Matplotlib style to apply to the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance
        Figure object containing the pointing errors plot.
    """
    az_res, el_res = coo_meas.spherical_offsets_to(coo_ref)
    with plt.style.context(style, {'xtick.labelsize': 16, 'ytick.labelsize': 16}):
        matplotlib.rcParams.update({'font.size': 14})
        x = coo_ref.az
        y = 90 * u.degree - coo_ref.alt  # use zenith angle here as a trick
        uu = (az_res).to(u.arcsec).value
        vv = (-1 * el_res).to(u.arcsec).value
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=[8, 8])
        qq = ax.quiver(
            x.to(u.radian).value,
            y.value,
            uu,
            vv,
            np.sqrt(uu**2 + vv**2),
            scale_units='y',
            angles='xy',
            pivot='tip',
            color='red'
        )
        ax.set_rmax(90)
        ticks = [0, 15, 30, 45, 60, 75, 90]
        ax.set_rticks(ticks)
        ax.set_rlim(0, 90)
        ax.set_yticklabels([
            r"$90^{\circ}$",
            r"$75^{\circ}$",
            r"$60^{\circ}$",
            r"$45^{\circ}$",
            r"$30^{\circ}$",
            r"$15^{\circ}$",
            r"El = $0^{\circ}$",
        ])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.scatter(x.to(u.radian).value, y.value)
        cbar = plt.colorbar(qq, shrink=0.7)
        cbar.set_label("arcsec")
        plt.tight_layout()
    return fig


def pointing_azel_resid(coo_ref, coo_meas, style="ggplot"):
    """
    Plot of az/el pointing residuals as a function of az/el

    Parameters
    ----------
    coo_ref : `~astropy.coordinates.SkyCoord` instance
        Reference coordinates
    coo_mes : `~astropy.coordinates.SkyCoord` instance
        Measured coordinates
    style : str (default: ggplot)
        Matplotlib style to apply to the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance
        Figure object containing the az/el residuals plot.
    """
    az_res, el_res = coo_meas.spherical_offsets_to(coo_ref)
    az_res = az_res.to(u.arcsec)
    el_res = el_res.to(u.arcsec)
    az = coo_ref.az
    el = coo_ref.alt
    azel_max = np.max([5, np.abs(az_res).max().to(u.arcsec).value, np.abs(el_res).max().to(u.arcsec).value])
    with plt.style.context(style, {'xtick.labelsize': 18, 'ytick.labelsize': 18}):
        matplotlib.rcParams.update({'font.size': 16})
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex='col', sharey='row')

        axs[0, 0].set_ylim(-azel_max, azel_max)
        axs[1, 0].set_xlim(0, 360)
        axs[1, 0].set_ylim(-azel_max, azel_max)
        axs[1, 1].set_xlim(0, 90)

        axs[0, 0].scatter(az, az_res)
        axs[1, 0].scatter(az, el_res)

        axs[0, 1].scatter(el, az_res)
        axs[1, 1].scatter(el, el_res)

        axs[0, 0].set_ylabel(r"$\Delta A$ (arcsec)")
        axs[1, 0].set_ylabel(r"$\Delta E$ (arcsec)")
        axs[1, 0].set_xlabel("Azimuth")
        axs[1, 0].set_xticks([0, 90, 180, 270, 360])
        axs[1, 0].set_xticklabels(["N", "E", "S", "W", "N"])
        axs[1, 1].set_xlabel("Elevation")
        axs[1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
        axs[1, 1].set_xticklabels([f"{i}" + r"$^{\circ}$" for i in [0, 15, 30, 45, 60, 75, 90]])

    plt.tight_layout()
    return fig


def plot_corner(idata, quantiles=None, truths=None, title_kwargs={"fontsize": 18}, label_kwargs={"fontsize": 16}):
    """
    Make corner plot from outputs of a pymc az/el fit

    Parameters
    ----------
    idata : object
        Any object that can be converted to an `~arviz.InferenceData` object. Refer to documentation of
        arviz.convert_to_dataset for details.
    quantiles : list of float (default: None -> [0.16, 0.5, 0.84])
        Quantiles to overlay on each histogram plot
    truths : dict
        Dict of reference parameters to overlay on plots. Must contain the following keys:
        ia, ie, an, aw, ca, npae, tf, tx, el_sigma, and az_sigma.
        Values set to None won't be displayed. Default is to not display any.

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance
        Figure object containing the corner plot.
    """
    if truths is None:
        truths = {}

    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    pars = best_fit_pars(idata)
    labels = []
    for p in pars.keys():
        if p not in truths:
            truths[p] = None

        if 'sigma' in p:
            ax, _ = p.split('_')
            labels.append("$\\sigma_{" + ax.upper() + "}$")
        else:
            labels.append(p.upper())

    fig = corner.corner(
        idata,
        labels=labels,
        quantiles=quantiles,
        truths=truths,
        show_titles=True,
        title_kwargs=title_kwargs,
        label_kwargs=label_kwargs
    )
    return fig


def plot_posterior(idata):
    """
    Make posterior probability distributions plot from a pymc fit

    Parameters
    ----------
    idata : object
        Any object that can be converted to an `~arviz.InferenceData` object.
        Refer to documentation of arviz.convert_to_dataset for details.

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance
        Figure object containing the corner plot.
    """
    fig = arviz.plot_posterior(idata)
    return fig
