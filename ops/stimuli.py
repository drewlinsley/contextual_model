#!/usr/bin/env python

import numpy as np
import scipy as sp
from model_utils import mul, ifloor, iceil, iround

DEFAULT_FDOMAIN = (0., 1.)
DEFAULT_SVAL = .375#214
DEFAULT_CVAL = .625#271
DEFAULT_CVAL2 = .375#214
DEFAULT_CONCENTRIC_VALUES = sp.linspace(0.375, 0.625, 6)

#------------------------------------------------------------------------------#
def normalize_stimulus_range(stimulus, native_range, target_range):
    """ Docstring for normalize_stimulus_range
    """
    
    stimulus = (stimulus - native_range[0])/(native_range[1] - native_range[0])
    stimulus = stimulus * (target_range[1] - target_range[0]) + target_range[0]

    return stimulus

#------------------------------------------------------------------------------#
def get_center_nfsurrounds(size=101, csize=1, nsize=9, fsize=29,
    cval=None, nval=None, fval=None, bgval=sp.nan):
    """
    """

    assert size >= fsize >= nsize >= csize

    a = lambda s: size//2 - ifloor(s/2.)
    b = lambda s: size//2 + iceil(s/2.)

    f_ar = sp.zeros((fsize, fsize)) + fval
    n_ar = sp.zeros((nsize, nsize)) + nval
    c_ar = sp.zeros((csize, csize)) + cval

    stimulus = sp.zeros((size, size)) + bgval
    stimulus[a(fsize):b(fsize), a(fsize):b(fsize)] = f_ar
    stimulus[a(nsize):b(nsize), a(nsize):b(nsize)] = n_ar
    stimulus[a(csize):b(csize), a(csize):b(csize)] = c_ar

    return stimulus

#------------------------------------------------------------------------------#
def get_center_surround(size=101, csize=9, sval=DEFAULT_SVAL, cval=DEFAULT_CVAL,
    normalize_range=False, native_range=DEFAULT_FDOMAIN,
    target_range=DEFAULT_FDOMAIN, checkerboard=False, cval2=DEFAULT_CVAL2):
    """ Docstring for get_center_surround
    """

    a = int(size//2 - sp.floor(csize/2.))
    b = int(size//2 + sp.ceil(csize/2.))

    stimulus = sp.zeros((size, size)) + sval
    center_stim = sp.zeros((csize, csize)) + cval

    if checkerboard:
        center_stim.ravel()[1::2] = cval2
    stimulus[a:b, a:b] = center_stim

    if normalize_range:
        stimulus = normalize_stimulus_range(stimulus,
            native_range=native_range, target_range=target_range)

    return stimulus

#------------------------------------------------------------------------------#
def get_concentric(size=101, values=DEFAULT_CONCENTRIC_VALUES, annular_width=3,
    bg=sp.nan, normalize_range=False, native_range=DEFAULT_FDOMAIN,
    target_range=DEFAULT_FDOMAIN):
    """ Docstring for get_concentric
    """

    nvalues, vmin, vmax = len(values), values.min(), values.max()
    stimulus = sp.zeros((size, size))
    stimulus[:] = bg

    for i, v in enumerate(values[::-1]):
        r = annular_width * (nvalues - i)
        stimulus[size//2-r:size//2+r+1, size//2-r:size//2+r+1] = v

    if normalize_range:
        stimulus = normalize_stimulus_range(stimulus,
            native_range=native_range, target_range=target_range)
    
    return stimulus

#------------------------------------------------------------------------------#
def westheimer(sz, h, w, valc, valc2, vals, sepc, seps):
    """"""

    stim = sp.zeros((sz, sz))
    stim[:] = sp.nan
    a = sz//2-int(sp.floor(h/2.0))
    b = sz//2+int(sp.ceil(h/2.0))
    c = sz//2-int(sp.floor(w/2.0))
    d = sz//2+int(sp.ceil(w/2.0))

    stim[a:b, c:d] = valc # central bar
    stim[a:b, c-w-sepc:d-w-sepc] = valc2 # left-to-central bar
    stim[a:b, c+w+sepc:d+w+sepc] = valc2 # right-to-central bar
    stim[a:b, c-2*w-sepc-seps:d-2*w-sepc-seps] = vals # left flanker
    stim[a:b, c+2*w+sepc+seps:d+2*w+sepc+seps] = vals # right flanker

    return stim
    
#------------------------------------------------------------------------------#
def westheimer_simple(sz, h, w, cval, sval, sep):
    """"""

    stim = sp.zeros((sz, sz))
    stim[:] = sp.nan
    a = sz//2-int(sp.floor(h/2.0))
    b = sz//2+int(sp.ceil(h/2.0))
    c = sz//2-int(sp.floor(w/2.0))
    d = sz//2+int(sp.ceil(w/2.0))

    stim[a:b, c:d] = cval # central bar
    stim[a:b, c-w-sep:d-w-sep] = sval # left-to-central bar
    stim[a:b, c+w+sep:d+w+sep] = sval # right-to-central bar

    return stim

#------------------------------------------------------------------------------#
