import numpy as np
from astropy import units

import coord
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
import lsst.afw.image as afw_image
from descwl_shear_sims.sim.constants import ZERO_POINT
from lsst.afw.math import FixedKernel
from lsst.meas.algorithms import KernelPsf
from lsst.daf.base import PropertyList


def make_exps(band_data, show=False):
    """
    make lsst stack exposures for each image and noise image
    """

    from lsst.afw.cameraGeom.testUtils import DetectorWrapper

    exps = []
    psf_exps = []
    byband_exps = {}
    byband_psf_exps = {}

    for band in band_data:
        bdata = band_data[band]

        byband_exps[band] = []
        byband_psf_exps[band] = []

        for epoch_ind, se_obs in enumerate(bdata):

            wcs = se_obs.wcs
            pos = wcs.toImage(wcs.center)
            # pos = wcs.center

            image = se_obs.image.array
            bmask = se_obs.bmask.array
            weight = se_obs.weight.array

            weight = se_obs.weight.array

            psf_gsimage, psf_offset = se_obs.get_psf(
                pos.x,
                pos.y,
                center_psf=False,
                get_offset=True,
            )
            psf_image = psf_gsimage.array

            # TODO: deal with zeros
            w = np.where(weight > 0)
            assert w[0].size == weight.size
            noise_var = 1.0/weight

            sy, sx = image.shape

            masked_image = afw_image.MaskedImageF(sx, sy)
            masked_image.image.array[:, :] = image
            masked_image.variance.array[:, :] = noise_var
            masked_image.mask.array[:, :] = bmask

            # an exp for the PSF
            # we need to put a var here that is consistent with the
            # main image to get a consistent coadd.  I'm not sure
            # what the best choice is, using median for now TODO
            pny, pnx = psf_image.shape
            pmasked_image = afw_image.MaskedImageF(pny, pnx)
            pmasked_image.image.array[:, :] = psf_image
            pmasked_image.variance.array[:, :] = np.median(noise_var)
            pmasked_image.mask.array[:, :] = 0

            exp = afw_image.ExposureF(masked_image)
            psf_exp = afw_image.ExposureF(pmasked_image)

            zperr = 0.0
            calib_mean = (ZERO_POINT*units.ABmag).to_value(units.nJy)
            calib_err = (np.log(10.) / 2.5) * calib_mean * zperr
            calib = afw_image.PhotoCalib(calib_mean, calib_err)

            exp.setPsf(make_stack_psf(psf_image))
            exp.setWcs(make_stack_wcs(wcs))
            exp.setPhotoCalib(calib)

            psf_exp.setWcs(make_stack_psf_wcs(
                dims=psf_gsimage.array.shape,
                jac=psf_gsimage.wcs,
                offset=psf_offset,
                world_origin=wcs.center,
            ))

            detector = DetectorWrapper().detector
            exp.setDetector(detector)
            psf_exp.setDetector(detector)

            if show:
                from espy import images
                images.view(exp.image.array)
                # show_image_and_mask(exp)
                # input('hit a key')

            exps.append(exp)
            byband_exps[band].append(exp)

            psf_exps.append(psf_exp)
            byband_psf_exps[band].append(psf_exp)

    return exps, psf_exps, byband_exps, byband_psf_exps


def make_stack_psf(psf_image):
    """
    make fixed image psf for stack usage
    """
    return KernelPsf(
        FixedKernel(
            afw_image.ImageD(psf_image.astype(float))
        )
    )


def make_stack_wcs(wcs):
    """
    convert galsim tan wcs to stack wcs
    """

    if wcs.wcs_type == 'TAN':
        crpix = wcs.crpix
        stack_crpix = geom.Point2D(crpix[0], crpix[1])
        cd_matrix = wcs.cd

        crval = geom.SpherePoint(
            wcs.center.ra/coord.radians,
            wcs.center.dec/coord.radians,
            geom.radians,
        )
        stack_wcs = makeSkyWcs(
            crpix=stack_crpix,
            crval=crval,
            cdMatrix=cd_matrix,
        )
    elif wcs.wcs_type == 'TAN-SIP':
        import galsim

        # this is not used if the lower bounds are 1, but the extra keywords
        # GS_{X,Y}MIN are set which we will remove below

        fake_bounds = galsim.BoundsI(1, 10, 1, 10)
        hdr = {}
        wcs.writeToFitsHeader(hdr, fake_bounds)

        del hdr["GS_XMIN"]
        del hdr["GS_YMIN"]

        metadata = PropertyList()

        for key, value in hdr.items():
            metadata.set(key, value)

        stack_wcs = makeSkyWcs(metadata)

    return stack_wcs


def make_stack_psf_wcs(*, dims, offset, jac, world_origin):
    """
    convert the galsim jacobian wcs to stack wcs
    for a tan projection

    Parameters
    ----------
    dims: (ny, nx)
        dims of the psf
    offset: seq or array
        xoffset, yoffset
    jac: galsim jacobian
        From wcs
    world_origin: origin of wcs
        get from coadd_wcs.center
    """
    import galsim

    cy, cx = (np.array(dims)-1)/2
    cy += offset.y
    cx += offset.x
    origin = galsim.PositionD(x=cx, y=cy)

    tan_wcs = galsim.TanWCS(
        affine=galsim.AffineTransform(
            jac.dudx, jac.dudy, jac.dvdx, jac.dvdy,
            origin=origin,
            world_origin=galsim.PositionD(0, 0),
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )

    return make_stack_wcs(tan_wcs)
