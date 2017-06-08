#!/usr/bin/env python

""" Authors: David A. Mely <david_mely@brown.edu>
             Thomas Serre <thomas_serre@brown.edu>
"""

from __future__ import absolute_import
try:
    import colored_traceback.always
except ImportError:
    pass
import h5py
import time
import itertools as it
from copy import deepcopy
import numpy as np
from numpy import r_, s_
import scipy as sp
from hmax.models.hnorm import models as mod
from hmax.models.hnorm.barams import floatX
import sklearn as sl
from sklearn import utils
from sklearn import linear_model
from sklearn import preprocessing

# graphics packages and settings
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable as maxl
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['axes.grid'] = False
default_cmap = 'viridis'
plt.style.use('ggplot')

# color imports
import skimage as si
from skimage import img_as_float
from skimage import color
from skimage.color import xyz2rgb
from skimage.color import rgb2xyz
from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
from skimage.color import xyz2lab
from skimage.color import lab2xyz
from skimage.color import xyz2luv
from skimage.color import luv2xyz


#------------------------------------------------------------------------------#
# Single-Opponent model
#######################

# default LN-SO model with standard channels (no weird stuff like Red-Cyan...)
_DEFAULT_SO_RFSIZE = 9
_DEFAULT_SO_CHANNELS = [0, 1, 3, 4, 5, 7]#[0, 1, 4, 5]
_DEFAULT_SO_PARAMETERS = {
    'filters': {'name': 'gabors', 'aspect_ratio': .6,
    'sizes': sp.array([_DEFAULT_SO_RFSIZE]),
    'spatial_frequencies': sp.array([[9.0]]),
    'orientations': sp.arange(2)*sp.pi/2, 'phases': sp.array([0]),
    'with_center_surround': False, 'padding': 'reflect',
    'corr': False, 'ndp': False},
    'model': {'channels_so': ('R+G-', 'B+Y-', 'R+C-', 'Wh+Bl-',
        'G+R-', 'Y+B-', 'C+R-', 'Bl+Wh-'),'normalize': False},
    'dnp_so': None, 'selected_channels': _DEFAULT_SO_CHANNELS,
    'norm_channels': _DEFAULT_SO_CHANNELS}

_CS_SO_RFSIZE = 5
_CS_SO_PARAMETERS = {
    'filters': {'name': 'gabors', 'aspect_ratio': .6,
    'sizes': sp.array([_CS_SO_RFSIZE]),
    'spatial_frequencies': sp.array([[5.0]]),
    'orientations': sp.arange(2)*sp.pi/2, 'phases': sp.array([0]),
    'with_center_surround': True, 'padding': 'reflect',
    'corr': False, 'ndp': False},
    'model': {'channels_so': ('R+G-', 'B+Y-', 'R+C-', 'Wh+Bl-',
        'G+R-', 'Y+B-', 'C+R-', 'Bl+Wh-'),'normalize': False},
    'dnp_so': None, 'selected_channels': _DEFAULT_SO_CHANNELS,
    'norm_channels': _DEFAULT_SO_CHANNELS}


CCPAR = {
    'continuous': False,
}

def GET_SO_CS(im_lms):
    """"""
    assert im_lms.ndim == 3
    assert im_lms.shape[-1] == 3
    im_lms = sp.array(im_lms, dtype=floatX)
    im_lms = sp.rollaxis(im_lms, -1, 0)
    sel = _CS_SO_PARAMETERS['selected_channels']
    so = mod.ventral_so(p=_CS_SO_PARAMETERS, image=im_lms, verbose=False)
    return so.squeeze()[sel, -1, :, :]

def GET_SX_CS(im_lms):
    """"""
    so = sp.array([GET_SO_CS(im_lms)])

def GET_SO_STANDARD(im_lms):
    """"""
    assert im_lms.ndim == 3
    assert im_lms.shape[-1] == 3
    im_lms = sp.array(im_lms, dtype=floatX)
    im_lms = sp.rollaxis(im_lms, -1, 0)
    # im_lms = sp.array(im_lms.swapaxes(0, -1), dtype=floatX)
    sel = _DEFAULT_SO_PARAMETERS['selected_channels']
    so = mod.ventral_so(p=_DEFAULT_SO_PARAMETERS, image=im_lms, verbose=False)
    return so.squeeze().max(-3)[sel, :, :]


#------------------------------------------------------------------------------#
# Miscellaneous XYZ, RGB, LMS conversions
#########################################

# CIE 2002 Color Appearance Model
CIECAM02_CAT = sp.array([[ 0.7328,  0.4296, -0.1624],
                         [-0.7036,  1.6975,  0.0061],
                         [ 0.003 ,  0.0136,  0.9834]])

def XYZ2xyY(XYZimage):
    """"""
    xyYimage = sp.zeros_like(XYZimage)
    xyYimage[..., [0]] = XYZimage[..., [0]] / XYZimage.sum(-1, keepdims=True)
    xyYimage[..., [1]] = XYZimage[..., [1]] / XYZimage.sum(-1, keepdims=True)
    xyYimage[..., [2]] = XYZimage[..., [1]]

    # if RGB = 0, 0, 0, then manage division by zero
    xyYimage[XYZimage.sum(-1) == 0, :] = \
        sp.array([0.0, 0.0, 0.0]).reshape((1,) * (xyYimage.ndim - 1) + (3,))
        
    return xyYimage

def xyY2XYZ(xyYimage):
    """"""
    XYZimage = sp.zeros_like(xyYimage)
    XYZimage[..., [0]] = xyYimage[..., [2]]/xyYimage[..., [1]]*xyYimage[..., [0]]
    XYZimage[..., [1]] = xyYimage[..., [2]]
    XYZimage[..., [2]] = xyYimage[..., [2]]/xyYimage[..., [1]]* \
        (1 - xyYimage[..., [0]] - xyYimage[..., [1]])
    return XYZimage

def XYZ2lms(XYZimage):
    """"""
    return sp.tensordot(XYZimage, CIECAM02_CAT, axes=(-1, 1))

def lms2rgb(lmsimage):
    """"""
    return xyz2rgb(sp.tensordot(lmsimage,
        sp.linalg.inv(CIECAM02_CAT), axes=(-1, 1)))

def rgb2lms(rgbimage):
    """"""
    return XYZ2lms(rgb2xyz(rgbimage))

def xyz2hsv(XYZimage):
    """"""
    return rgb2hsv(xyz2rgb(XYZimage))

def hsv2xyz(HSVimage):
    """"""
    return rgb2xyz(hsv2rgb(HSVimage))

def delta_angle(angle1, angle2):
    """Takes the difference between two angles
    and return a value between -pi and pi."""
    return sp.arctan2(sp.sin(angle1 - angle2), sp.cos(angle1 - angle2))


#------------------------------------------------------------------------------#
# MacLeod-Boynton (lsY) color space
###################################
""" Re-used the formulae from PsychToolBox-3:
https://github.com/Psychtoolbox-3/Psychtoolbox-3/blob/ ...
... master/Psychtoolbox/PsychColorimetric/LMSToMacBoyn.m
"""

_k_MLB = sp.array([0.6373, 0.3924, 1.0])

def LMS2lsY(LMSimage):
    """ LMS -> MacLeod-Boynton """
    kLMSimage = LMSimage * _k_MLB.reshape((1, 1, 3)) # rescaled version
    lsYimage = sp.zeros_like(kLMSimage)
    lsYimage[..., 0] = kLMSimage[..., 0]/(kLMSimage[..., 0]+kLMSimage[..., 1])
    lsYimage[..., 1] = kLMSimage[..., 2]/(kLMSimage[..., 0]+kLMSimage[..., 1])
    lsYimage[..., 2] = kLMSimage[..., 0]+kLMSimage[..., 1]
    return lsYimage

def lsY2LMS(lsYimage):
    """ MacLeod-Boynton -> LMS """
    LMSimage = sp.zeros_like(lsYimage)
    LMSimage[..., 0] = lsYimage[..., 0] * lsYimage[..., 2]
    LMSimage[..., 1] = lsYimage[..., 2] - LMSimage[..., 0]
    LMSimage[..., 2] = lsYimage[..., 1] * lsYimage[..., 2]
    LMSimage /= _k_MLB.reshape((1, 1, 3))
    return LMSimage

#------------------------------------------------------------------------------#
# Derrington-Krauskopf-Lennie (DKL) color space
###############################################
def cart2sph(xyz):
    """ Radius, azimuth follow the canonical definition.
    Elevation is the angle in [-pi, pi], with 0 being the
    isoluminant plane (z = 0 in cartesian coordinates).
    """
    rae = sp.zeros_like(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    rae[..., 0] = sp.sqrt(x**2 + y**2 + z**2)           # radius
    rae[..., 1] = sp.arctan2(y, x)                      # azimuth
    rae[..., 2] = sp.arctan2(z, sp.sqrt(x**2 + y**2))   # elevation
    return rae

def sph2cart(rae):
    """"""
    xyz = sp.zeros_like(rae)
    rd, az, el = rae[..., 0], rae[..., 1], rae[..., 2]
    xyz[..., 0] = rd * sp.cos(el) * sp.cos(az)
    xyz[..., 1] = rd * sp.cos(el) * sp.sin(az)
    xyz[..., 2] = rd * sp.sin(el)
    return xyz

def build_DKL_space(Y=0.75):
    """ Given some reference luminance point Y, build the DKL color space
    (in Cartesian coordinates) following the prescription of Brainard, 1996
    [NB: Brainard, 1996 has a typo in an important formula!]
    """

    gray_xyY = sp.array([0.310, 0.316, Y])
    gray_lms = XYZ2lms(xyY2XYZ(gray_xyY)).ravel()
    BK1 = -gray_lms[0]/gray_lms[1]
    BK2 = (gray_lms[0]+gray_lms[1])/gray_lms[2]
    k_LM = 0.3727   # dkl[LM]   := L - M
    k_S = 0.1667    # dkl[S]    := S - (Lum)
    k_LUM = 0.2887  # dkl[Lum]  := L + M
    LMS2DKL = sp.array([[ k_LM,     k_LM*BK1,       0.0],
                        [-k_S,     -k_S,        k_S*BK2],
                        [ k_LUM,    k_LUM,          0.0]])
    DKL2LMS = sp.linalg.inv(LMS2DKL) # plane isoluminant to gray bg is z = 0

    def lms2dklC(lms_array):
        """ DKL := (L-M [red-green], S-(L+M) [blue-yellow], L+M [luminance])
        """
        # get cone differences w.r.t. the background
        diff_array = lms_array.copy()
        diff_array[..., 0] -= gray_lms[0]
        diff_array[..., 1] -= gray_lms[1]
        diff_array[..., 2] -= gray_lms[2]
        dkl_array = sp.tensordot(diff_array, LMS2DKL, axes=((-1,), (1,)))
        return dkl_array

    def dklC2lms(dlkC_array):
        """"""
        diff_array = sp.tensordot(dlkC_array, DKL2LMS, axes=((-1,), (1,)))
        diff_array[..., 0] += gray_lms[0]
        diff_array[..., 1] += gray_lms[1]
        diff_array[..., 2] += gray_lms[2]
        return diff_array

    def rgb2dklC(rgb_array):
        """"""
        return lms2dklC(rgb2lms(rgb_array))

    def dklC2rgb(dlkC_array):
        """"""
        return lms2rgb(dklC2lms(dlkC_array))

    return lms2dklC, dklC2lms, rgb2dklC, dklC2rgb

def get_isoluminant_colors_DKL(Y, r, n):
    """ Y: luminance in CIE 1993 XYZ
        r: radius in DKL space (roughly a saturation parameter)
    """

    # generate the DKL space at that Y
    _, _, _, dklC2rgb = build_DKL_space(Y=Y)

    # generate colors isoluminant to zero elevation in that DKL space
    r_dklS = sp.zeros((n, 1, 1, 1)) + r
    a_dklS = sp.linspace(-sp.pi, sp.pi, n).reshape((n, 1, 1, 1))
    e_dklS = sp.zeros((n, 1, 1, 1))
    isolum_dklS = sp.concatenate((r_dklS, a_dklS, e_dklS), axis=-1)
    isolum_dklC = sph2cart(isolum_dklS)
    isolum_rgb = sp.array([dklC2rgb(im) \
        for im in isolum_dklC]).reshape((n, 1, 1, 3))

    return isolum_rgb

#------------------------------------------------------------------------------#
def get_dataset(nh, ns, nv, sz, randomize=True, bg_hsv=None, bg_sz=None,
    patch_type='center_surround'):
    """"""

    assert (bg_hsv is None) == (bg_sz is None)

    im_rgb = sp.zeros((nh, ns, nv, sz, sz, 3))
    im_hsv = sp.zeros((nh, ns, nv, sz, sz, 3))
    im_XYZ = sp.zeros((nh, ns, nv, sz, sz, 3))
    im_lms = sp.zeros((nh, ns, nv, sz, sz, 3))
    hs = sp.linspace(0.00, 1.00, nh)
    ss = sp.linspace(0.20, 0.50, ns)
    vs = sp.linspace(0.60, 1.00, nv)

    if bg_sz is not None:
        im_hsv[..., :] = sp.array(bg_hsv).ravel()

    for idx, h in enumerate(hs):
        for jdx, s in enumerate(ss):
            for kdx, v in enumerate(vs):
                if bg_sz is not None:
                    if patch_type == 'center_surround':
                        im_hsv[idx, jdx, kdx,
                            sz//2-bg_sz//2:sz//2+bg_sz//2+1,
                            sz//2-bg_sz//2:sz//2+bg_sz//2+1] \
                            = sp.array([[[h, s, v]]])
                    elif patch_type == 'line':
                        im_hsv[idx, jdx, kdx, sz//2, :] \
                            = sp.array([h, s, v])
                else:
                    im_hsv[idx, jdx, kdx] = sp.array([[[h, s, v]]])
                im_rgb[idx, jdx, kdx] = si.color.hsv2rgb(im_hsv[idx, jdx, kdx])
                im_XYZ[idx, jdx, kdx] = si.color.rgb2xyz(im_rgb[idx, jdx, kdx])

    im_lms = XYZ2lms(im_XYZ)
    im_rgb.shape = (nh*ns*nv, sz, sz, 3)
    im_hsv.shape = (nh*ns*nv, sz, sz, 3)
    im_XYZ.shape = (nh*ns*nv, sz, sz, 3)
    im_lms.shape = (nh*ns*nv, sz, sz, 3)

    if randomize:
        im_rgb, im_hsv, im_XYZ, im_lms = sl.utils.shuffle(
            im_rgb, im_hsv, im_XYZ, im_lms)

    return im_rgb, im_hsv, im_XYZ, im_lms

#------------------------------------------------------------------------------#
def get_Xy(fget, images_LMS, images_XYZ):
    """"""

    X = []
    n, h, w, _ = images_LMS.shape

    pbar = pb(n, 'Computing color features on %i images' % (n,))
    for idx, im in enumerate(images_LMS):
        X.append(fget(im)[:, h//2, w//2])
        pbar.update(idx)
    pbar.finish()

    y = images_XYZ[:, h//2, w//2, :]

    return sp.array(X), y

#------------------------------------------------------------------------------#
def get_XYZ2RGB_predictor(reg_X, reg_Y, reg_Z, scaler):
    """"""

    def predict_image(nkhw, rescale=True, out='rgb'):
        """"""
        n_, k_, h_, w_ = nkhw.shape
        nkhw_ = nkhw.swapaxes(1, 0).reshape((k_, n_*h_*w_)).T

        if rescale:
            nkhw_ = scaler.transform(nkhw_)

        im = sp.zeros((n_, h_, w_, 3))
        X_ = reg_X.predict(nkhw_).reshape((n_, h_*w_))
        Y_ = reg_Y.predict(nkhw_).reshape((n_, h_*w_))
        Z_ = reg_Z.predict(nkhw_).reshape((n_, h_*w_))

        for idx, (x_, y_, z_) in enumerate(zip(X_, Y_, Z_)):
            XYZ = sp.array([x_, y_, z_]).T.reshape((h_, w_, 3))
            if out == 'rgb':
                im[idx] = si.color.xyz2rgb(XYZ)
            elif out == 'hsv':
                im[idx] = xyz2hsv(XYZ)
            elif out == 'lab':
                im[idx] = si.color.xyz2lab(XYZ)
            elif out == 'luv':
                im[idx] = si.color.xyz2luv(XYZ)
            elif out == 'xyz':
                im[idx] = XYZ
            elif out == 'lms':
                im[idx] = XYZ2lms(XYZ)
            elif out == 'lsY':
                im[idx] = LMS2lsY(XYZ2lms(XYZ))
            elif out == 'dkl':
                im[idx] = None
            else:
                raise ValueError('Output image format not recognized')

        return im

    return predict_image

