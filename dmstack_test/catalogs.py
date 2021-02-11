import galsim
from descwl_shear_sims.sim.shifts import get_shifts


class ColorGalaxyCatalog(object):
    """
    Galaxies of fixed galsim type, flux, and size

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        dimensions of the coadd
    buff: int
        Buffer region with no objects, on all sides of image
    layout: string
        The layout of objects, either 'grid' or 'random'
    mag: float
        Magnitude of all objects
    hlr: float
        Half light radius of all objects
    """
    def __init__(self, *, rng, coadd_dim, buff, layout, imag, hlr, gal_type):
        self._gal_type = gal_type
        # for the sim bug
        self.gal_type = 'exp'
        self.hlr = hlr
        self.rng = rng

        self.gmr = 1.4
        self.rmi = 0.6
        self.imz = 0.35

        self.mags = {}
        self.mags['i'] = imag
        self.mags['r'] = self.rmi + self.mags['i']
        self.mags['g'] = self.gmr + self.mags['r']
        self.mags['z'] = self.mags['i'] - self.imz

        self.shifts = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout=layout,
        )

        if gal_type == 'dev':
            self.gal_class = galsim.DeVaucouleurs
        elif gal_type == 'exp':
            self.gal_class = galsim.Exponential
        else:
            raise ValueError('gal type should be exp or dev')

    def get_objlist(self, *, survey, g1, g2):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects]
        """

        mag = self.mags[survey.filter_band]
        flux = survey.get_flux(mag)

        num = self.shifts.size
        objlist = [
            self._get_galaxy(i, flux).shear(g1=g1, g2=g2)
            for i in range(num)
        ]

        shifts = self.shifts.copy()
        return objlist, shifts

    def _get_galaxy(self, i, flux):
        """
        get a galaxy object

        Parameters
        ----------
        i: int
            Index of object
        flux: float
            Flux of object

        Returns
        --------
        galsim.GSObject
        """
        return self.gal_class(
            half_light_radius=self.hlr,
            flux=flux,
        ).shift(
            dx=self.shifts['dx'][i],
            dy=self.shifts['dy'][i]
        )
