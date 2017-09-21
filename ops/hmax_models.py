#!/usr/bin/env python

"""
Models of the ventral stream and the dorsal stream, including divisive normalization.
"""

import sys
import scipy as sp
import numpy as np
import copy
sys.path.append('/home/drew/Documents/hmax/models/hnorm')
from computation_legacy import flexfilter, hwrectify

# LEGACY
from hmax.models.hnorm.utils import get_pooled_array_size_listed
from hmax.tools.io2 import allocator
from interface import *
from templates import *

#-------------------------------------------------------------------------------------------------#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OLD BACKEND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def divnorm(dnp, ninarray, dinarray, verbose=True):
    """
    A parametric form for the divisive normalization model. Supports "local"
    pooling (a.k.a. convolution-like, calls flexfilter), "global" pooling
    (e.g., sum or max, calls flexpooling), or None (doesn't do anything).
    WARNING: the pool parsing does not support any kind of tree!
    """

    # Load parameters for the divisive normalization equation
    if dnp is None:
        print('| No divisive normalization '+ \
            'applied: empty parameter dictionary.')
        return ninarray
    else:
        npp = deepcopy(dnp['npool'])
        dpp = deepcopy(dnp['dpool'])
        dpn = dnp['numeric']
        sigma = dpn['sigma']
        q1 = dpn['q1']
        q2 = dpn['q2']
        p = dpn['p']
        r = dpn['r']

    # Generally, pools can be iterable (then sum over these before normalizing)
    try:
        no_npool = npp['type'] is None
        npp = [npp]
    except TypeError:
        no_npool = all([nkw['type'] is None for nkw in npp])
    try:
        no_dpool = dpp['type'] is None
        dpp = [dpp]
    except TypeError:
        no_dpool = all([dkw['type'] is None for dkw in dpp])
    if no_npool and no_dpool:
        return ninarray
    # Predict output shapes
    noutsize = get_pooled_array_size_listed(
        npp, ninarray.shape, name='npool')
    doutsize = get_pooled_array_size_listed(
        dpp, dinarray.shape, name='dpool')
    noutarray = allocator(noutsize)
    doutarray = allocator(doutsize)

    # Compute pooled unit activities: numerator
    if verbose: pbar = pb(len(npp), 'Divnorm: pooling numerator')
    for i, nkw in enumerate(npp):
        noutarray += recursive_pool(
            ninarray**p, params=nkw, keyname='npool', verbose = verbose)
        if verbose: pbar.update(i)
    if verbose: pbar.finish()

    # Compute pooled unit activities: denominator
    if verbose: pbar = pb(len(dpp), 'Divnorm: pooling denominator')
    for i, dkw in enumerate(dpp):
        doutarray += recursive_pool(
            dinarray**q1, params=dkw, keyname='dpool', verbose = verbose)
        if verbose: pbar.update(i)
    if verbose: pbar.finish()

    # Normalize numerator pool by denominator pool
    if verbose: print('| Divnorm: normalizing')
    normalized_array = noutarray/((sigma ** 2 + doutarray ** q2) ** r)

    if sp.isnan(normalized_array).any():
        import ipdb; ipdb.set_trace()

    return normalized_array



# @_support_input_resize_and_crop
def dorsal_velocity(p, image, crop=False, resize=False, precomputed=None, handle=None):
    """ OUTPUT DIMENSIONS : [SPEED, DIRECTION, T, H, W]
    Model of velocity-selective cells in the medio-temporal of the dorsal stream (MT).
    Such cells pool from V1 afferents (computed with dorsal_primary).
    """

    from computation import flexfilter

    # Parameters for divisive normalization
    P = copy.deepcopy(p)
    dnp_velocity, P['dnp'] = P.pop('dnp_velocity'), P.pop('dnp_V1')
    no = p['filters']["gabors_number_of_orientations"]
    ndp = p['filters']['ndp']
    inhibition = P['model']['inhibition']
    vectorform = P['model']['vectorform']
    nv = len(p['model']['speeds'])

    # Get the primary (V1-like) responses first
    print('| ---- Calling V1 model.')
    st = dorsal_primary(P, image, crop=crop, resize=resize, handle=handle) \
        if precomputed is None else precomputed

    # Preallocation
    h, w = st.shape[-2:]
    h_, w_ = get_size_after_downsampling(dnp_velocity, h , w) \
        if inhibition == 'divisive' else (h, w)
    fshape = (nv, no, st.shape[-3], h_, w_)
    mt = allocator(fshape, handle=handle, name='m')

    # Divisive normalization & velocity pooling
    print('| ---- Calling MT model.')
    if inhibition == 'subtractive':
        posweights = dnp_velocity['npool']['ndfilter']
        negweights = dnp_velocity['dpool']['ndfilter']
        pos = flexfilter(posweights, st[:, :, :no, ...],
            axes=(-3, -2, -1), ndp=ndp)
        neg = flexfilter(negweights, st[:, :, :no, ...],
            axes=(-3, -2, -1), ndp=ndp)
        mt[...] = hwrectify(pos - neg, '+', 1)
    elif inhibition == 'divisive':
        mt[...] = divnorm(dnp_velocity,
            st[:, :, :no, ...], st[:, :, :no, ...])
    else:
        raise Exception('Unknown inhibition type. Please choose from: <subtractive, divisive>.')

    # Positive speeds, directions in [0, 2Pi[
    if vectorform:
        i1, i2 = map(int, (sp.floor(nv/2.), sp.ceil(nv/2.)))
        static = [i1] if i1 != i2 else []
        mt = sp.concatenate([sp.concatenate([mt[static]] * 2, axis=1), \
            sp.concatenate([mt[:i1][::-1], mt[i2:]], axis=1)], axis=0)

    return st, mt

#-------------------------------------------------------------------------------------------------#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OLD BACKEND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# @_support_input_resize_and_crop
def dorsal_primary(p, image, crop=False, resize=False, handle=None):
    """ OUTPUT DIMENSIONS : [SCALE, TEMPORAL FREQUENCY, ORIENTATION, T, H, W]
    Model of motion-sensitive cells in the primary areas of the dorsal stream (V1).
    Such receptive fields are band-pass in the spectral frequency domain.
    """

    #:: from computation import flexfilter
    from hmax.backend._theano_.conv import conv_3d

    # Pre-process the image
    #:: im = preprocess_image(p, image)
    im = get_image(image, color=False, crop=crop, resize=resize)

    # Parameters
    nf = p['filters']["gabors_number_of_frames"]
    omegas = p['filters']["gabors_temporal_frequencies"]
    withcs = p['filters']["with_center_surround"]
    no = p['filters']["gabors_number_of_orientations"]
    np = len(p['filters']["gabors_phase"])
    ns = len(p['filters']["gabors_sizes"])
    nw = len(omegas)
    nfim, h, w = im.shape[0], im.shape[-2], im.shape[-1]
    conv3d = nf != nfim
    dnp = p['dnp']
    ndp = p['filters']['ndp']

    # Either convolve along temporal dimension or sum and collapse
    if conv3d:
        axes = (-3, -2, -1)
        fshape = (ns, nw, no+int(withcs), nfim, h, w)
    else:
        axes = (-2, -1)
        fshape = (ns, nw, no+int(withcs), 1, h, w)
        im = im[::-1, :, :]

    # Generate the filters and preallocate
    gabors = get_gabors_spatiotemporal(p)
    st = allocator(fshape, handle=handle, name='v')

    # Filter
    for s in range(ns):
        #:: message = 'Dorsal primary -- scale %i out of %i' % (s+1, ns)
        #:: out = flexfilter(gabors[s], im, axes=axes, mode='edge',
        #::   ndp=ndp, message=message)
        out = conv_3d(gabors[s], im, axes=axes, padding='edge',
            ndp=ndp, verbose=True, im_div='auto')
        if not conv3d:
            out = sp.expand_dims(out, -3)
        st[s] = energy(out, axis=0)

    # Normalization, excluding center-surround
    st[:, :, :no, ...] = divnorm(dnp, st[:, :, :no, ...], st[:, :, :no, ...])

    return st

#-------------------------------------------------------------------------------------------------#
def ventral_absolute_disparity(p, image, handle=None):
    """ OUTPUT DIMENSIONS : [ABSOLUTE DISPARITY, SCALE, ORIENTATION, H, W]
    Model of absolute disparity tuning in the ventral stream
    If not normalize: produces broadly-tuned cells known as:
        TE:     tuned-excitatory
        TI:     tuned-inhibitory
    If normalize: produces cells tuned to absolute disparity
    """

    # Parameters
    im = preprocess_image(p, image)
    ns = len(p['filters']["gabors_sizes"])
    withcs = p['filters']["with_center_surround"]
    no = p['filters']["gabors_number_of_orientations"]
    np = len(p['filters']["gabors_phase"])
    disparities = p['model']["disparities"]
    nd = len(disparities)
    normalize = p['model']['normalize']
    ndp = p['filters']['ndp']
    dnp_V1 = p['dnp_V1']
    dnp_disparity = p['dnp_disparity']
    h, w = im.shape[-2], im.shape[-1]
    h_, w_ = h, w
    invalid = sp.ceil(sp.absolute(disparities).max()/2.)
    invalid_ = get_size_after_downsampling(dnp_disparity, invalid, invalid)[0]
    h_, w_ = get_size_after_downsampling(dnp_disparity, h, w)
    h_, w_ = (h_, w_ - 2 * invalid_)
    if not normalize:
        te_pooling = copy.deepcopy(dnp_disparity['npool'])
        ti_pooling = copy.deepcopy(dnp_disparity['dpool']['dpool'])
        te_pool = globals()[te_pooling.pop('type')]
        ti_pool = globals()[ti_pooling.pop('type')]

    # Preallocate
    fshape = (nd, ns, no+int(withcs), h_, w_)
    if normalize:
        ad = allocator(fshape, handle=handle, name='ad')
    else:
        te = allocator(fshape, handle=handle, name='te')
        ti = allocator(fshape, handle=handle, name='ti')
    pte = sp.zeros((ns, np, no+int(withcs), 2, h, w))
    pti = sp.zeros((ns, np, no+int(withcs), 2, h, w))

    # Monocular responses
    g = get_gabors_binocular(p)
    pbar = start_progressbar(4*ns, 'Binocular receptive fields')
    for s in range(ns):
        pte[s][:, :, 0] = flexfilter(g['TE'][s][:, :, 0], im[0], ndp=ndp, verbose=False)
        pte[s][:, :, 1] = flexfilter(g['TE'][s][:, :, 1], im[1], ndp=ndp, verbose=False)
        pti[s][:, :, 0] = flexfilter(g['TI'][s][:, :, 0], im[0], ndp=ndp, verbose=False)
        pti[s][:, :, 1] = flexfilter(g['TI'][s][:, :, 1], im[1], ndp=ndp, verbose=False)
        update_progressbar(pbar, 4*s)
    end_progressbar(pbar, 'Processed %i scales.'  % (ns,))
    pte = sp.concatenate((hwrectify(pte, '+', 1), hwrectify(pte, '-', 1)), axis=1)
    pti = sp.concatenate((hwrectify(pti, '+', 1), hwrectify(pti, '-', 1)), axis=1)

    # Weaken boundaries w.r.t. textures (d.n. within orientation across scale bands)
    print('| V1-level normalization across scales and phases, within orientation bands')
    pte = divnorm(dnp_V1, pte, pte)
    pti = divnorm(dnp_V1, pti, pti)

    # Binocular responses
    pbar = start_progressbar(nd, 'Tuned-excitatory/tuned-inhibitory cells')
    for i, d in enumerate(disparities):
        te_ = energy(sp.sum(shiftimage(pte, d, stereo_axis=-3), axis=-3), axis=1)
        ti_ = energy(sp.sum(shiftimage(pti, d, stereo_axis=-3), axis=-3), axis=1)
        if normalize:
            ad[i] = divnorm(dnp_disparity, te_, ti_, verbose=False)[..., invalid_:-invalid_]
        else:
            te[i] = te_pool(nimage=te_, **te_pooling)[..., invalid_:-invalid_]
            ti[i] = ti_pool(nimage=ti_, **ti_pooling)[..., invalid_:-invalid_]
        update_progressbar(pbar, i)
    end_progressbar(pbar, 'Processed %i absolute disparities.'  % (nd,))

    return ad if normalize else [te, ti]

#-------------------------------------------------------------------------------------------------#
# @_support_input_resize_and_crop
def ventral_do(p, image, handle=None):
    """ OUTPUT DIMENSIONS : [SCALE, COLOR CHANNEL, ORIENTATION, H, W]
    Model of double-opponent color-tuned cells in the ventral stream
    """

    # Pre-process the image
    im = preprocess_image(p, image)
    h, w = im.shape[-2], im.shape[-1]

    # Parameters
    ns = len(p['filters']["gabors_sizes"])
    cs = p['filters']["with_center_surround"]
    no = p['filters']["gabors_number_of_orientations"]
    nc = len(p['model']['channels'])
    np = len(p['filters']["gabors_phase"])
    ndp_so = p['filters']['ndp_so']
    ndp_do = p['filters']['ndp_do']
    dnp_so = p['dnp_so']
    dnp_do = p['dnp_do']

    # Generate the filters and preallocate
    sz = (ns, nc, no+int(cs), h, w)
    do = allocator(sz, handle=handle, name='do')
    so = sp.zeros((ns, 2*nc, no+int(cs), h, w))
    sop = sp.zeros(sz)
    son = sp.zeros(sz)
    sg = get_gabors_color(p)
    dg = get_gabors(p)

    # Make the SO Gabors monophase
    sg = [g[:, 0, ...] for g in sg]

    # Single-opponent stage: filtering + opponent divisive normalization
    for s in range(ns):
        out = flexfilter(sg[s], im, ndp=ndp_so,
            message='Single-opponent stage -- scale %i out of %i' % (s+1, ns))
        sop[s] = hwrectify(out, '+', 1)
        son[s] = hwrectify(out, '-', 1)
    so[:, :nc, :no] = divnorm(dnp_so, sop[:, :, :no], son[:, :, :no])
    so[:, nc:, :no] = divnorm(dnp_so, son[:, :, :no], sop[:, :, :no])

    # Double-opponent stage
    for s in range(ns):
        print('| Double-opponent stage: scale %i out of %i' % (s+1, ns))
        pbar = start_progressbar(nc*(no+int(cs)), 'Double-opponent filtering')
        for c in range(nc):
            for o in range(no+int(cs)):
                pos = flexfilter(dg[s][:, o, ...], so[s, c, o],
                    ndp=ndp_do, verbose=False)
                neg = flexfilter(dg[s][:, o, ...], so[s, c+nc, o],
                    ndp=ndp_do, verbose=False)
                do[s, c, o] = energy(sp.sqrt(pos**2 + neg**2), axis=0)
                update_progressbar(pbar, c*(no+int(cs))+o)
        end_progressbar(pbar,
            'Processed 2 dimensions in a total of %i iterations.' % (nc*(no+int(cs))))

    # Divisive normalization
    do[..., :no, :, :] = divnorm(dnp_do, do[..., :no, :, :], do[..., :no, :, :])

    return do

#-------------------------------------------------------------------------------------------------#
# @_support_input_resize_and_crop
def ventral_so(p, image, handle=None, verbose=False):
    """ OUTPUT DIMENSIONS : [SCALE, COLOR CHANNEL, ORIENTATION, H, W]
    Model of single-opponent color-tuned cells in the ventral stream
    """

    def sampler(x):
        # return abs(np.random.uniform(low=x - 0, high=x + 0) + x) ** np.random.uniform(low=1.,high=1.)  # previously did [0, 2]
        # return abs(np.random.uniform(low=x - 1, high=x + 1) + x) ** np.random.uniform(low=-2.,high=2.)  # previously did [0, 2]
        # return np.sign(np.random.uniform() - 0.5) * abs(np.random.uniform(low=x - 1, high=x + 1) + x) ** -2  # previously did [0, 2]
        return np.random.uniform(low=-.5, high=.5)

    # Pre-process the image
    # im = preprocess_image(p, image)
    im = image

    # Parameters
    h, w = im.shape[-2], im.shape[-1]
    ns = len(p['filters']["sizes"])
    withcs = p['filters']["with_center_surround"]
    no = len(p['filters']["orientations"])
    nc = len(p['model']['channels_so'])
    ndp = p['filters']['ndp']

    # Parameters for divisive normalization
    dnp = p['dnp_so']

    # Generate the filters and preallocate
    sz = (ns, nc, no+int(withcs), h, w)
    so = allocator(sz, handle=handle, name='so')
    g = get_gabors_color(p)

    # Filter
    for s in range(ns):
        out = flexfilter(g[s], im, ndp=ndp)
        so[s] = hwrectify(out + sampler(0), '+', 1)
        # so[s] = sp.concatenate((hwrectify(out, '+', 1), hwrectify(out, '-', 1)), axis=0)

    # Normalization: Mely's canonical form
    so[:] = divnorm(dnp, so[:], so[:])
    # so_a = so[:, :nc/2, :no, ...].copy()
    # so_b = so[:, nc/2:, :no, ...].copy()
    # so[:, :nc/2, :no, ...] = divnorm(dnp, so_a, so_b)
    # so[:, nc/2:, :no, ...] = divnorm(dnp, so_b, so_a)

    return so

#-------------------------------------------------------------------------------------------------#
# @_support_input_resize_and_crop
def ventral_legacy(p, image, handle=None):
    """ OUTPUT DIMENSIONS : [SCALE, PHASE, ORIENTATION, H, W]
    Model of contrast-, spatial frequency-tuned cells in the ventral stream
    """

    # Pre-process the image
    im = preprocess_image(p, image)

    # Parameters
    h, w = im.shape[-2], im.shape[-1]
    ns = len(p['filters']["gabors_sizes"])
    withcs = p['filters']["with_center_surround"]
    no = p['filters']["gabors_number_of_orientations"]
    np = len(p['filters']["gabors_phase"])
    combine_phases = p['model']['combine_phases']
    ndp = p['filters']['ndp']

    # Parameters for divisive normalization
    dnp = p['dnp']

    # Generate the filters and preallocate
    gabors = get_gabors(p)
    features = allocator((ns, np, no+int(withcs), h, w), handle=handle)

    from hmax.backend._scipy_.conv import conv_nd as flexfilter
    K = {'axes': (-2, -1), 'mode': 'constant', 'method': 'fft', 'verbose': True}

    # Filter
    for s in range(ns):
        features[s] = flexfilter(gabors[s], im, ndp=ndp, **K)

    # Divisive normalization
    features[..., :no, :, :] = divnorm(dnp, features[..., :no, :, :],
        energy(features[..., :no, :, :], axis=-4, keepdims=True))

    # Optional energy stage (phase-invariance)
    if combine_phases:
        features = energy(features[...], axis=-4, keepdims=False)

    return features
