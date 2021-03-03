import sys
import numpy as np
import logging
import ngmix
from ngmix.bootstrap import BDFBootstrapper, Bootstrapper
import metadetect

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
from .util import Namer
from .vis import show_sim

MEDS_CONFIG = {
    'box_type': 'sigma_size',
    'sigma_fac': 5,
    'min_box_size': 32,
    'max_box_size': 128,
    # 'stamps': {
    #     'min_stamp_size': 32,
    #     'max_stamp_size': 128,
    #     'sigma_factor': 5,
    #     'bits_to_ignore_for_weight': [],
    #     'bits_to_null': [],
    # }
}


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
    layout='grid',
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
        'layout': layout,
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

        mbobs = make_mbobs(band_data=sim_data['band_data'])

        toutput = run_ngmix(
            mbobs=mbobs,
            rng=rng, model=ngmix_model,
        )

        dlist.append(toutput)

    truth = np.zeros(1, dtype=[
        ('gal_type', 'S10'),
        ('true_imag', 'f8'),
        ('true_hlr', 'f8')
    ])
    truth['gal_type'] = gal_type
    truth['true_imag'] = gal_imag
    truth['true_hlr'] = gal_hlr

    data = eu.numpy_util.combine_arrlist(dlist)
    logger.info('writing: %s' % output)
    with fitsio.FITS(output, 'rw', clobber=True) as fits:
        fits.write(data, extname='model_fits')
        fits.write(truth, extname='truth')


def get_jac(wcs, cenx, ceny):
    """
    get jacobian at the coadd image center, and make
    an ngmix jacobian with center specified (this is not the
    position used to evaluate the jacobian)
    """
    import galsim

    crpix = wcs.crpix
    galsim_pos = galsim.PositionD(x=crpix[0], y=crpix[1])

    galsim_jac = wcs.jacobian(image_pos=galsim_pos)

    return ngmix.Jacobian(
        x=cenx,
        y=ceny,
        dudx=galsim_jac.dudx,
        dudy=galsim_jac.dudy,
        dvdx=galsim_jac.dvdx,
        dvdy=galsim_jac.dvdy,
    )


def make_mbobs(*, band_data):
    mbobs = ngmix.MultiBandObsList()
    for band, bdata in band_data.items():
        obslist = ngmix.ObsList()

        for epoch_ind, se_obs in enumerate(bdata):
            ny, nx = se_obs.image.array.shape
            cenx = (nx - 1)/2
            ceny = (ny - 1)/2
            jacobian = get_jac(wcs=se_obs.wcs, cenx=cenx, ceny=ceny)

            psf_gsimage = se_obs.get_psf(
                cenx,
                ceny,
                center_psf=False,
            )
            psf_image = psf_gsimage.array

            psf_cen = (np.array(psf_image.shape)-1)/2
            psf_jac = jacobian.copy()
            psf_jac.set_cen(row=psf_cen[0], col=psf_cen[1])

            psf_noise_fake = psf_image.max()/50000
            psf_image += rng.normal(scale=psf_noise_fake, size=psf_image.shape)
            psf_weight = psf_image*0 + 1.0/psf_noise_fake**2

            psf_obs = ngmix.Observation(
                image=psf_image,
                weight=psf_weight,
                jacobian=psf_jac,
            )

            obs = ngmix.Observation(
                image=se_obs.image.array,
                weight=se_obs.weight.array,
                jacobian=jacobian,
                psf=psf_obs,
            )
            obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


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


def run_ngmix(*, mbobs, model, rng):
    medsifier = metadetect.detect.MEDSifier(
        mbobs=mbobs,
        sx_config=None,
        meds_config=MEDS_CONFIG,
    )
    mbm = medsifier.get_multiband_meds()

    mbobs_list = mbm.get_mbobs_list(weight_type='uberseg')

    # old bootstrapper
    psf_Tguess = ngmix.moments.fwhm_to_T(0.8)  # noqa
    max_pars = {
        'method': 'lm',
        'lm_pars': {
            'maxfev': 4000,
            'ftol': 1e-05,
            'xtol': 1e-05,
        }
    }

    n = Namer(front=model)

    prior = get_prior(rng=rng, model=model)
    nband = len(mbobs)
    output = make_output(
        nband=nband, num=len(mbobs_list), ngmix_model=model,
    )
    assert nband == 1, '1 band for now'

    for i, mbobs in enumerate(mbobs_list):
        try:

            if model == 'bdf':
                boot = BDFBootstrapper(mbobs)
            else:
                boot = Bootstrapper(mbobs)

            boot.fit_psfs(
                # 'em3',
                'coellip3',
                psf_Tguess*rng.uniform(low=0.9, high=1.1),
            )
            scale = mbobs[0][0].jacobian.scale

            boot.fit_gal_psf_flux()
            pres = boot.get_psf_flux_result()
            output['psf_flags'][i] = pres['flags'][0]
            output['psf_flux'][i] = pres['psf_flux'][0]/scale**2
            output['psf_flux_err'][i] = pres['psf_flux_err'][0]/scale**2

            output['psf_mag'][i] = (
                ZERO_POINT - 2.5*np.log10(output['psf_flux'][i])
            )

            if model == 'bdf':
                boot.fit_max(
                    max_pars,
                    prior=prior,
                    ntry=2,
                )
            else:
                boot.fit_max(
                    model,
                    max_pars,
                    prior=prior,
                    ntry=2,
                )

            res = boot.max_fitter.get_result()
            output[n('flags')][i] = res['flags']
            if res['flags'] == 0:
                scale = mbobs[0][0].jacobian.scale
                output[n('flux')][i] = res['flux'] / scale**2
                output[n('flux_err')][i] = res['flux_err'] / scale**2

                output[n('mag')][i] = (
                    ZERO_POINT - 2.5*np.log10(output[n('flux')][i])
                )
                if model == 'bdf':
                    output['bdf_fracdev'][i] = res['pars'][5]

        except ngmix.gexceptions.BootPSFFailure:
            output[n('flags')][i] = 2**29
        except ngmix.gexceptions.BootGalFailure:
            output[n('flags')][i] = 2**30

    return output


def get_prior(*, model, rng):
    cen_prior = ngmix.priors.CenPrior(0.0, 0.0, 0.2, 0.2, rng=rng)
    g_prior = ngmix.priors.GPriorBA(0.2, rng=rng)
    T_prior = ngmix.priors.FlatPrior(-0.1, 1.e+05, rng=rng)  # noqa
    flux_prior = ngmix.priors.FlatPrior(-1000.0, 1.0e+09, rng=rng)

    if model == 'bdf':
        fracdev_prior = ngmix.priors.Normal(
            0.5, 0.1,
            bounds=[0.0, 1.0],
            rng=rng,
        )

        prior = ngmix.joint_prior.PriorBDFSep(
            cen_prior,
            g_prior,
            T_prior,
            fracdev_prior,
            flux_prior,
        )
    else:
        prior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            flux_prior,
        )

    return prior


def make_output(*, nband, num, ngmix_model):

    if nband == 1:
        bshape = ()
    else:
        bshape = (nband, 1)

    n = Namer(front=ngmix_model)
    dt = [
        ('psf_flags', 'i4') + bshape,
        ('psf_flux', 'f8') + bshape,
        ('psf_flux_err', 'f8') + bshape,
        ('psf_mag', 'f8') + bshape,
        (n('flags'), 'i4'),
        (n('flux'), 'f8') + bshape,
        (n('flux_err'), 'f8') + bshape,
        (n('mag'), 'f8') + bshape,
    ]
    if ngmix_model == 'bdf':
        dt += ('bdf_fracdev', 'f8'),

    data = np.zeros(num, dtype=dt)
    for name in data.dtype.names:
        if 'flags' not in name:
            data[name] = np.nan

    return data
