import numpy as np


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front == '':
            front = None
        if back == '' or back == 'noshear':
            back = None

        self.front = front
        self.back = back

        if self.front is None and self.back is None:
            self.nomod = True
        else:
            self.nomod = False

    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)

        return n


def make_truth(cat, bands):
    truth = np.zeros(1, dtype=[
        ('gal_type', 'S10'),
        ('bands', 'U1', len(bands)),
        ('true_gmag', 'f8'),
        ('true_rmag', 'f8'),
        ('true_imag', 'f8'),
        ('true_zmag', 'f8'),
        ('true_gmr', 'f8'),
        ('true_rmi', 'f8'),
        ('true_imz', 'f8'),
        ('true_hlr', 'f8')
    ])
    truth['bands'][0] = bands
    truth['gal_type'] = cat.gal_type
    truth['true_gmag'] = cat.mags['g']
    truth['true_rmag'] = cat.mags['r']
    truth['true_imag'] = cat.mags['i']
    truth['true_zmag'] = cat.mags['z']
    truth['true_hlr'] = cat.hlr
    truth['true_gmr'] = cat.gmr
    truth['true_rmi'] = cat.rmi
    truth['true_imz'] = cat.imz

    return truth
