import numpy as np
import lsst.afw.image as afw_image
from lsst.afw.cameraGeom.testUtils import DetectorWrapper


def make_simple_coadd(*, exps):
    """
    assume wcs, psf is all the same
    """

    sy, sx = exps[0].image.array.shape
    image = np.zeros((sy, sx))
    weight = np.zeros((sy, sx))

    wsum = 0.0
    for exp in exps:
        tweight = 1.0/exp.variance.array

        wt = np.median(tweight)
        image += exp.image.array[:, :]*wt
        weight += tweight

        wsum += wt

    image *= 1.0/wsum

    noise_var = 1.0/weight

    masked_image = afw_image.MaskedImageF(sx, sy)
    masked_image.image.array[:, :] = image
    masked_image.variance.array[:, :] = noise_var
    masked_image.mask.array[:, :] = exps[0].mask.array

    coadd_exp = afw_image.ExposureF(masked_image)

    coadd_exp.setPsf(exps[0].getPsf())
    coadd_exp.setWcs(exps[0].getWcs())

    detector = DetectorWrapper().detector
    coadd_exp.setDetector(detector)
    return coadd_exp
