def show_image_and_mask(exp):
    """
    show the image and mask in ds9, with
    mask colored

    Parameters
    ----------
    exp: afw_image.MaskedImageF
        The image to show
    """
    import lsst.afw.display as afw_display
    display = afw_display.getDisplay(backend='ds9')
    display.mtv(exp)
    display.scale('log', 'minmax')


def show_sim(data):
    """
    show an image
    """
    from espy import images

    imlist = []
    for band in data:
        for se_obs in data[band]:
            sim = se_obs.image.array
            sim = images.asinh_scale(image=sim/sim.max(), nonlinear=0.14)
            imlist.append(sim)
            imlist.append(se_obs.get_psf(25.1, 31.5).array)

    images.view_mosaic(imlist, dpi=150, colorbar=True)
