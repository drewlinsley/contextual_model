from hmax.tools.utils import pb, mul, ifloor, iceil, iround
import scipy as sp

def get_wl87_stim(size, dists, cval, sval, ch, cw, sh, sw):
    """"""
    im = sp.zeros((len(dists), 1, size, size))
    im[:] = sp.nan
    for idx, ds in enumerate(dists):
        u = size//2 - ifloor(ch/2.)
        v = size//2 + iceil(ch/2.)
        w = size//2 - ifloor(cw/2.)
        z = size//2 + iceil(cw/2.)
        a = size//2 - ifloor(sw/2.)
        b = size//2 + iceil(sw/2.)
        it = u - ds - sh
        jt = v + ds
        im[idx, 0, u:v, w:z] = cval
        im[idx, 0, it:it+sh, a:b] = sval
        im[idx, 0, jt:jt+sh, a:b] = sval

    return im


