import scipy as sp
import model_cutils as cutils
from hmax.tools.utils import pb, mul, ifloor, iceil, iround


# general settings
##################
_DEFAULT_SM2003_WORKDIR = \
    '/home/dmely/src/hmax/models/ucircuits/contextual/working'
_DEFAULT_SM2003_VIEW_CENTER_ONLY = True

# if True, get the so activities by full SO model with convolution.
# otherwise, insert appropriate population vectors at each pixel (default)
# the latter option is more or less equivalent to assuming RF size to be 1 pixel
_DEFAULT_SM2003_CONVOLUTION = False

# helper functions
##################
def _adjust_Y(color_set, Y):
    """ Send an RGB triplet in xyY space, replace luminance Y,
    and retrieve back into RGB space again.
    """
    adjusted_set = {}
    for k, v in color_set.iteritems():
        new_rgb = cutils.xyz2rgb(cutils.xyY2XYZ(sp.array(
            [[cutils.XYZ2xyY(cutils.rgb2xyz(sp.array(
            [[color_set[k]]]))).ravel().tolist()[:-1] + [Y]]])))
        adjusted_set[k] = tuple(new_rgb.ravel().tolist())
    return adjusted_set

def _get_responses_3regions(size, csize, nsize, fsize,
    cval, nval, fval, bgval=None):
    """ Generate activation values """

    assert len(cval.ravel()) == len(nval.ravel()) == len(fval.ravel())
    nc = len(cval.ravel())
    im = sp.zeros((nc, size, size))
    if bgval is not None:
        im[:, :, :] = bgval[:].reshape((nc, 1, 1))
    im[:, \
        size//2-ifloor(fsize/2.):size//2+iceil(fsize/2.), \
        size//2-ifloor(fsize/2.):size//2+iceil(fsize/2.)] \
        = fval.ravel().reshape((nc, 1, 1))
    im[:, \
        size//2-ifloor(nsize/2.):size//2+iceil(nsize/2.), \
        size//2-ifloor(nsize/2.):size//2+iceil(nsize/2.)] \
        = nval.ravel().reshape((nc, 1, 1))
    im[:, \
        size//2-ifloor(csize/2.):size//2+iceil(csize/2.), \
        size//2-ifloor(csize/2.):size//2+iceil(csize/2.)] \
        = cval.ravel().reshape((nc, 1, 1))

    return im

def _get_cyclical_stimuli_concentric(sz, csz, ssz, cps):
    """"""
    assert sz % 2 == 1
    assert cps > 0
    assert cps <= sz//2
    assert (ssz is None) or (ssz % 2 == 1)
    gx, gy = sp.mgrid[-sz//2+1:sz//2+1, -sz//2+1:sz//2+1]
    gr = sp.maximum(sp.absolute(gx), sp.absolute(gy))
    bounds = sp.linspace(0, sz//2, cps+1)
    for idx in range(cps):
        gr[(bounds[idx] < gr) * (gr <= bounds[idx+1])] = \
            (-1 if (idx % 2) else -2)
    gr[gr == -2] = 1
    gr[sz//2-csz//2:sz//2+csz//2+1, sz//2-csz//2:sz//2+csz//2+1] = 0
    if ssz is not None:
        gr = gr.astype(float)
        gr[sp.maximum(sp.absolute(gx), sp.absolute(gy)) > ssz//2] = sp.nan
    return gr

def _get_cyclical_stimuli_linear_old(sz, csz, ssz, cps):
    """"""
    assert sz % 2 == 1
    assert cps > 0
    assert cps <= sz//2
    assert (ssz is None) or (ssz % 2 == 1)
    gx, _ = sp.mgrid[-sz//2+1:sz//2+1, -sz//2+1:sz//2+1]
    gr = sp.absolute(gx)
    bounds = sp.linspace(0, sz//2, cps+1)
    for idx in range(cps):
        gr[(bounds[idx] < gr) * (gr <= bounds[idx+1])] = \
            (-1 if (idx % 2) else -2)
    gr[gr == -2] = 1
    gr[sz//2-csz//2:sz//2+csz//2+1, :] = 0
    if ssz is not None:
        gr = gr.astype(float)
        gr[sp.absolute(gx) > ssz//2] = sp.nan
    return gr

def _get_cyclical_stimuli_linear(sz, csz, ssz, cps):
    """"""
    assert sz % 2 == 1
    assert cps > 0
    assert cps <= sz//2
    assert (ssz is None) or (ssz % 2 == 1)
    gx, _ = sp.mgrid[-sz//2+1:sz//2+1, -sz//2+1:sz//2+1]
    gr = sp.absolute(gx)
    bounds = sp.linspace(csz//2, sz//2, cps+1)
    for idx in range(cps):
        gr[(bounds[idx] < gr) * (gr <= bounds[idx+1])] = \
            (-1 if (idx % 2) else -2)
    gr[gr == -2] = 1
    gr[sz//2-csz//2:sz//2+csz//2+1, :] = 0
    if ssz is not None:
        gr = gr.astype(float)
        gr[sp.absolute(gx) > ssz//2] = sp.nan
    return gr

def _map_responses_on_cycles(size, csize, ssize, cps,
    cval, nval, fval, emptyval=None):
    """ Generate stimuli closer to Shevell & Monnier """

    assert len(cval.ravel()) == len(nval.ravel()) == len(fval.ravel())

    nc = len(cval.ravel())
    im = sp.zeros((nc, size, size))
    jm = _get_cyclical_stimuli_linear(size, csize, ssize, cps)

    im[:, jm ==  0] = cval.ravel().reshape((nc, 1))
    im[:, jm ==  1] = nval.ravel().reshape((nc, 1))
    im[:, jm == -1] = fval.ravel().reshape((nc, 1))
    if emptyval is not None:
        im[:, sp.isnan(jm)] = emptyval.ravel((nc, 1))
    else:
        im[:, sp.isnan(jm)] = 0.0

    return im

# stimulus color settings
#########################
_DEFAULT_SM2003_COLORS_RGB = {
    'orange':       sp.array([.99, .65, .47]),
    'purple':       sp.array([.65, .54, .77]),
        # adjusted to have Y=.30 #sp.array([.60, .49, .71]),
    'purple_alt':   sp.array([.60, .48, .70]),
    'lime':         sp.array([.56, .60, .49]),
        # adjusted to have Y=.30 #sp.array([.61, .65, .53]),
    'lime_alt':     sp.array([.51, .61, .53]),
    'turquoise':    sp.array([.44, .78, .73]),
    'neutral':      sp.array([.66, .66, .66]),
    'lilac':        sp.array([.89, .62, .80]),
    'citrus':       sp.array([.67, 1.0, .50]),
    'EEW':          sp.array([.58, .58, .58]),
        # adjusted to have Y=.30
}

# generate a bunch of isoluminant colors ...
Y_test_colors = .30
_, _, rgb2dklC, dklC2rgb = cutils.build_DKL_space(Y_test_colors)
purple_dklS = cutils.cart2sph(rgb2dklC(
    _DEFAULT_SM2003_COLORS_RGB['purple'].reshape((1, 1, 3))))
lime_dklS = cutils.cart2sph(rgb2dklC(
    _DEFAULT_SM2003_COLORS_RGB['lime'].reshape((1, 1, 3))))
r_test_colors = .03#(purple_dklS.squeeze()[0] + lime_dklS.squeeze()[0])/2.
n_test_colors = 25
isoluminant_test_colors = cutils.get_isoluminant_colors_DKL(
    Y=Y_test_colors, r=r_test_colors, n=n_test_colors)

# ... and append them to color dictionary
_CUSTOM_SM2003_COLORS_RGB_SAMPLED_IN_DKL = {
    '%02i' % i: isoluminant_test_colors[i].squeeze() \
        for i in range(n_test_colors)
}
_CUSTOM_SM2003_COLORS_RGB_SAMPLED_IN_HSV = {
    '%02i' % i: cutils.hsv2rgb( \
        sp.array([h, .28, .87]).reshape((1, 1, 3))).squeeze() \
        for i, h in enumerate(sp.linspace(0.0, 1.0, 25))
}
_DEFAULT_SM2003_COLORS_RGB.update(_CUSTOM_SM2003_COLORS_RGB_SAMPLED_IN_HSV)

# stimulus size settings
########################

# parameters from paper
_DEFAULT_SM2003_CSIZE_MIN = 6.0
_DEFAULT_SM2003_SIZE_MIN = 153.0
_MINPERPIX = 2.#alt::5. # how many minutes of visual angle per pixel

_DEFAULT_SM2003_CSIZE = iround(_DEFAULT_SM2003_CSIZE_MIN/_MINPERPIX) # <realistic>
_DEFAULT_SM2003_SIZE = \
     iround(_DEFAULT_SM2003_SIZE_MIN/_MINPERPIX) + \
    (iround(_DEFAULT_SM2003_SIZE_MIN/_MINPERPIX) % 2 == 0) # <realistic>
_DEFAULT_SM2003_SSIZE = None
_DEFAULT_SM2003_CXPAR = dict(cutils.CCPAR, srf=_DEFAULT_SM2003_CSIZE)

# _DEFAULT_SM2003_CSIZE = 1 # <works>
# _DEFAULT_SM2003_SIZE = 2**6 \
#     + _DEFAULT_SM2003_CSIZE \
#     + ((_DEFAULT_SM2003_CSIZE + 1) % 2) <works>

# default set of spatial frequencies for inducing surround stimuli
_DEFAULT_SM2003_CPDS = sp.array([0., 1., 2., 3.3, 5., 10.]) # <from paper>
_DEFAULT_SM2003_CPMS = _DEFAULT_SM2003_CPDS / 60.
_DEFAULT_SM2003_CPSS = iround(1 + _DEFAULT_SM2003_CPMS * \
    (_DEFAULT_SM2003_SIZE_MIN - _DEFAULT_SM2003_CSIZE_MIN)/2.0) # <realistic>
# _DEFAULT_SM2003_CPSS = 2 ** sp.arange(ifloor(
#     sp.log2(_DEFAULT_SM2003_SIZE//2)) + 1) <works>

# path to digitized experimental data
#####################################
_DEFAULT_SM2003_DATA_PREFIX = \
    '/home/dmely/src/hmax/models/ucircuits/contextual/data/'
_DEFAULT_SM2003_DATA_CSV = {
    'ObsML': {
        'PL': _DEFAULT_SM2003_DATA_PREFIX + 'SM2003_Fig5_ObsML_PL.csv',
        'LP': _DEFAULT_SM2003_DATA_PREFIX + 'SM2003_Fig5_ObsML_LP.csv',
    },
    'ObsPM': {
        'PL': _DEFAULT_SM2003_DATA_PREFIX + 'SM2003_Fig5_ObsPM_PL.csv',
        'LP': _DEFAULT_SM2003_DATA_PREFIX + 'SM2003_Fig5_ObsPM_LP.csv',
    },
}

# stimulus type for training set for regressors
REG_TRAIN_DATASET_PATCH_TYPE = 'center_surround'
