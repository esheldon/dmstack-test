import sys
import numpy as np
import logging

import fitsio
import esutil as eu

from descwl_shear_sims.sim import (
    make_sim,
    get_sim_config,
    make_psf,
    make_ps_psf,
    get_se_dim,
)
from descwl_shear_sims.sim.constants import ZERO_POINT

from .catalogs import ColorGalaxyCatalog
from .exposures import make_exps
from .run_stack import detect_and_deblend
from .run_ngmix import run_ngmix
from .util import Namer
from .vis import show_sim


def go(
    *,
    seed,
    gal_imag,
    gal_type,
    gal_hlr,
    ngmix_model,
    ntrial,
    output,
    show=False,
    loglevel='info',
):
    """
    seed: int
        Seed for a random number generator
    ntrial: int
        Number of trials to run, paired by simulation plus and minus shear
    output: string
        Output file path.  If output is None, this is a dry
        run and no output is written.
    show: bool
        If True, show some images.  Default False
    loglevel: string
        Log level, default 'info'
    """

    rng = np.random.RandomState(seed)

    g1, g2 = 0.0, 0.0

    # only over ride bands
    sim_config = {
        'bands': ["i"],
        'psf_type': 'moffat',
    }
    sim_config = get_sim_config(config=sim_config)

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger('mdet_lsst_sim')
    logger.setLevel(getattr(logging, loglevel.upper()))

    logger.info(str(sim_config))

    dlist = []

    for trial in range(ntrial):
        logger.info('-'*70)
        logger.info('trial: %d/%d' % (trial+1, ntrial))

        galaxy_catalog = ColorGalaxyCatalog(
            rng=rng,
            coadd_dim=sim_config['coadd_dim'],
            buff=sim_config['buff'],
            layout=sim_config['layout'],
            imag=gal_imag,
            hlr=gal_hlr,
            gal_type=gal_type,
        )

        if sim_config['psf_type'] == 'ps':
            se_dim = get_se_dim(coadd_dim=sim_config['coadd_dim'])
            psf = make_ps_psf(rng=rng, dim=se_dim)
        else:
            psf = make_psf(psf_type=sim_config["psf_type"])

        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=sim_config['coadd_dim'],
            g1=g1,
            g2=g2,
            psf=psf,
            psf_dim=sim_config['psf_dim'],
            dither=sim_config['dither'],
            rotate=sim_config['rotate'],
            bands=sim_config['bands'],
            epochs_per_band=sim_config['epochs_per_band'],
            noise_factor=sim_config['noise_factor'],
            cosmic_rays=sim_config['cosmic_rays'],
            bad_columns=sim_config['bad_columns'],
            star_bleeds=sim_config['star_bleeds'],
        )

        if show:
            show_sim(sim_data['band_data'])

        exps, psf_exps, byband_exps, byband_psf_exps = make_exps(
            band_data=sim_data['band_data'], show=show,
        )
        # coadd_exp = make_simple_coadd(exps=exps)

        # if show:
        #     from espy import images
        #     images.view(coadd_exp.image.array, plt_kws={'title': 'coadd'})

        # first just do i band as a test
        assert len(exps) == 1

        sources = detect_and_deblend(exp=exps[0], log=logger)

        toutput = make_output(num=len(sources), ngmix_model=ngmix_model)
        copy_sources_to_output(output=toutput, sources=sources)

        run_ngmix(
            sources=sources, exps=exps, output=toutput,
            rng=rng, model=ngmix_model,
        )

        dlist.append(toutput)

    truth = np.zeros(1, dtype=[
        ('gal_type', 'S10'),
        ('true_imag', 'f8'), ('true_hlr', 'f8')
    ])
    truth['gal_type'] = gal_type
    truth['true_imag'] = gal_imag
    truth['true_hlr'] = gal_hlr

    data = eu.numpy_util.combine_arrlist(dlist)
    logger.info('writing: %s' % output)
    with fitsio.FITS(output, 'rw', clobber=True) as fits:
        fits.write(data, extname='model_fits')
        fits.write(truth, extname='truth')


def make_output(*, num, ngmix_model):

    n = Namer(front=ngmix_model)
    dt = [
        ('psf_flags', 'i4'),
        ('psf_flux', 'f8'),
        ('psf_flux_err', 'f8'),
        ('psf_mag', 'f8'),

        ('cmodel_fracdev', 'f8'),
        ('cmodel_flags', 'i4'),
        ('cmodel_flux', 'f8'),
        ('cmodel_flux_err', 'f8'),
        ('cmodel_mag', 'f8'),

        ('cmodel_dev_fracdev', 'f8'),
        ('cmodel_dev_flags', 'i4'),
        ('cmodel_dev_flux', 'f8'),
        ('cmodel_dev_flux_err', 'f8'),
        ('cmodel_dev_mag', 'f8'),

        ('cmodel_exp_fracdev', 'f8'),
        ('cmodel_exp_flags', 'i4'),
        ('cmodel_exp_flux', 'f8'),
        ('cmodel_exp_flux_err', 'f8'),
        ('cmodel_exp_mag', 'f8'),

        (n('flags'), 'i4'),
        (n('flux'), 'f8'),
        (n('flux_err'), 'f8'),
    ]
    if ngmix_model == 'bdf':
        dt += ('bdf_fracdev', 'f8'),

    data = np.zeros(num, dtype=dt)
    for name in data.dtype.names:
        if 'flags' not in name:
            data[name] = np.nan

    return data


def copy_sources_to_output(*, output, sources):

    for i, rec in enumerate(sources):
        d = output[i]

        d['psf_flags'] = rec['base_PsfFlux_flag']
        d['psf_flux'] = rec['base_PsfFlux_instFlux']
        d['psf_flux_err'] = rec['base_PsfFlux_instFluxErr']

        if np.isnan(d['psf_flux']):
            d['psf_flags'] |= 2**30

        if d['psf_flags'] == 0:
            d['psf_mag'] = ZERO_POINT - 2.5*np.log10(d['psf_flux'])

        d['cmodel_flags'] = rec['modelfit_CModel_flag']

        d['cmodel_flux'] = rec['modelfit_CModel_instFlux']
        d['cmodel_flux_err'] = rec['modelfit_CModel_instFluxErr']

        if np.isnan(d['cmodel_flux']):
            d['cmodel_flags'] |= 2**30

        if d['cmodel_flags'] == 0:
            d['cmodel_mag'] = ZERO_POINT - 2.5*np.log10(d['cmodel_flux'])

        d['cmodel_dev_flags'] = rec['modelfit_CModel_dev_flag']
        d['cmodel_dev_flux'] = rec['modelfit_CModel_dev_instFlux']
        d['cmodel_dev_flux_err'] = rec['modelfit_CModel_dev_instFluxErr']

        if np.isnan(d['cmodel_dev_flux']):
            d['cmodel_dev_flags'] |= 2**30

        if d['cmodel_dev_flags'] == 0:
            d['cmodel_dev_mag'] = (
                ZERO_POINT - 2.5*np.log10(d['cmodel_dev_flux'])
            )

        d['cmodel_exp_flags'] = rec['modelfit_CModel_exp_flag']
        d['cmodel_exp_flux'] = rec['modelfit_CModel_exp_instFlux']
        d['cmodel_exp_flux_err'] = rec['modelfit_CModel_exp_instFluxErr']

        if np.isnan(d['cmodel_exp_flux']):
            d['cmodel_exp_flags'] |= 2**30

        if d['cmodel_exp_flags'] == 0:
            d['cmodel_exp_mag'] = (
                ZERO_POINT - 2.5*np.log10(d['cmodel_exp_flux'])
            )
