import ngmix
from ngmix.bootstrap import BDFBootstrapper, Bootstrapper
from metadetect.lsst_mbobs_extractor import MBObsExtractor

from .util import Namer


def run_ngmix(*, model, exps, sources, output, rng):
    meds_config = {
        'stamps': {
            'min_stamp_size': 32,
            'max_stamp_size': 128,
            'sigma_factor': 5,
            'bits_to_ignore_for_weight': [],
            'bits_to_null': [],
        }
    }
    mext = MBObsExtractor(
        config=meds_config,
        exposures=exps,
        sources=sources,
    )
    mbobs_list = mext.get_mbobs_list(weight_type='weight')

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
    for i, mbobs in enumerate(mbobs_list):
        try:

            if model == 'bdf':
                boot = BDFBootstrapper(mbobs)
            else:
                boot = Bootstrapper(mbobs)

            boot.fit_psfs(
                # 'em3',
                'coellip3',
                psf_Tguess,
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
                if model == 'bdf':
                    output['bdf_fracdev'][i] = res['pars'][5]

        except ngmix.gexceptions.BootPSFFailure:
            output[n('flags')][i] = 2**29
        except ngmix.gexceptions.BootGalFailure:
            output[n('flags')][i] = 2**30


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
