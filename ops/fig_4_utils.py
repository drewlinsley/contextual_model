import re, os
import numpy as np
import scipy as sp
from skimage import img_as_float
from skimage.color import xyz2rgb, rgb2xyz
from hmax.models.hnorm import models as mod
from hmax.tools.utils import pb
from hmax.models.ucircuits.contextual import stimuli as stim
from sklearn import linear_model

CIECAM02_CAT = sp.array([[ 0.7328,  0.4296, -0.1624],
                         [-0.7036,  1.6975,  0.0061],
                         [ 0.003 ,  0.0136,  0.9834]])

def xyY2XYZ(xyYimage):
    """"""
    XYZimage = sp.zeros_like(xyYimage)
    XYZimage[..., 0] = xyYimage[..., 2]/xyYimage[..., 1]*xyYimage[..., 0]
    XYZimage[..., 1] = xyYimage[..., 2]
    XYZimage[..., 2] = xyYimage[..., 2]/xyYimage[..., 1]* \
        (1 - xyYimage[..., 0] - xyYimage[..., 1])
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

def da2ha(a):
    """ Map angles from [0, 2*pi] -> [-pi, pi] """
    return sp.arctan2(sp.sin(a), sp.cos(a))

def GET_SO(im_lms,floatX,_DEFAULT_KW2015_SO_PARAMETERS):
    """"""
    assert im_lms.ndim == 3
    assert im_lms.shape[-1] == 3
    im_lms = sp.array(im_lms.swapaxes(0, -1), dtype=floatX)
    sel = _DEFAULT_KW2015_SO_PARAMETERS['selected_channels']
    so = mod.ventral_so(
        p=_DEFAULT_KW2015_SO_PARAMETERS,
        image=im_lms,
        verbose=False)
    return so.squeeze().max(-3)[sel, :, :]

def _regression_DKL(n_, fo_, fx_,stims_all_LM_, stims_all_S_, stims_all_LUM_):
    """ regress all coordinates but only hue matters """
    reg_fo_LM = linear_model.RidgeCV()
    reg_fo_S = linear_model.RidgeCV()
    reg_fo_LUM = linear_model.RidgeCV()
    reg_fx_LM = linear_model.RidgeCV()
    reg_fx_S = linear_model.RidgeCV()
    reg_fx_LUM = linear_model.RidgeCV()
    reg_fo_LM.fit(fo_[:n_], stims_all_LM_[:n_])
    reg_fo_S.fit(fo_[:n_], stims_all_S_[:n_])
    reg_fo_LUM.fit(fo_[:n_], stims_all_LUM_[:n_])
    reg_fx_LM.fit(fx_[:n_], stims_all_LM_[:n_])
    reg_fx_S.fit(fx_[:n_], stims_all_S_[:n_])
    reg_fx_LUM.fit(fx_[:n_], stims_all_LUM_[:n_])

    return reg_fo_LM, reg_fo_S, reg_fo_LUM, reg_fx_LM, reg_fx_S, reg_fx_LUM

# helper routine to display matrices of decoded colors
def _show_color_matrices(A_im, B_im, row_leg, col_leg, cell_sz=9, lw=1):
    """"""

    n_t_hues, n_s_hues, _ = A_im.shape
    i_sz = (cell_sz + lw) * (n_s_hues + 1)
    j_sz = (cell_sz + lw) * (n_t_hues + 1)
    A = sp.zeros((i_sz, j_sz, 3))
    B = sp.zeros((i_sz, j_sz, 3))

    # border regions (legend)
    A[:(cell_sz + lw), :(cell_sz + lw), :] = 1.0
    B[:(cell_sz + lw), :(cell_sz + lw), :] = 1.0

    for kdx in range(3):
        for idx in range(1, n_t_hues+1):
            cell = sp.zeros((cell_sz + lw, cell_sz + lw))
            cell[:-lw, :-lw] = row_leg[idx-1, kdx]
            cell[-lw:, :] = 1.0
            cell[:, -lw:] = 1.0
            A[idx*(cell_sz + lw):(idx+1)*(cell_sz + lw),
              :(cell_sz + lw), kdx] = cell
            B[idx*(cell_sz + lw):(idx+1)*(cell_sz + lw),
              :(cell_sz + lw), kdx] = cell

        for jdx in range(1, n_s_hues+1):
            cell = sp.zeros((cell_sz + lw, cell_sz + lw))
            cell[:-lw, :-lw] = col_leg[jdx-1, kdx]
            cell[-lw:, :] = 1.0
            cell[:, -lw:] = 1.0
            A[:(cell_sz + lw),
              jdx*(cell_sz + lw):(jdx+1)*(cell_sz + lw), kdx] = cell
            B[:(cell_sz + lw),
              jdx*(cell_sz + lw):(jdx+1)*(cell_sz + lw), kdx] = cell

    # fill main regions
    for kdx in range(3):
        for idx in range(1, n_t_hues+1):
            for jdx in range(1, n_s_hues+1):
                cell = sp.zeros((cell_sz + lw, cell_sz + lw))
                cell[:-lw, :-lw] = A_im[idx-1, jdx-1, kdx]
                cell[-lw:, :] = 1.0
                cell[:, -lw:] = 1.0
                A[idx*(cell_sz + lw):(idx+1)*(cell_sz + lw),
                  jdx*(cell_sz + lw):(jdx+1)*(cell_sz + lw),
                  kdx] = cell

        for idx in range(1, n_t_hues+1):
            for jdx in range(1, n_s_hues+1):
                cell = sp.zeros((cell_sz + lw, cell_sz + lw))
                cell[:-lw, :-lw] = B_im[idx-1, jdx-1, kdx]
                cell[-lw:, :] = 1.0
                cell[:, -lw:] = 1.0
                B[idx*(cell_sz + lw):(idx+1)*(cell_sz + lw),
                  jdx*(cell_sz + lw):(jdx+1)*(cell_sz + lw),
                  kdx] = cell

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(A, interpolation='none')
    ax[0].grid('off')
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].imshow(B, interpolation='none')
    ax[1].grid('off')
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    
    return fig, ax

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

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def create_stims(extra_vars):
    extra_vars = Bunch(extra_vars)
    out_dir = re.split('\.',extra_vars.f4_stimuli_file)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #################
    nc = len(extra_vars._DEFAULT_KW2015_SO_PARAMETERS['selected_channels'])

    # build stimuli
    ###############
    all_hues_dkls = sp.linspace(0.0, 2*sp.pi, extra_vars.n_train, endpoint=False)
    test_hues_dklS = all_hues_dkls[::extra_vars.n_train//extra_vars.n_t_hues][:extra_vars.n_t_hues]
    surr_hues_dklS = test_hues_dklS[::extra_vars.n_t_hues//extra_vars.n_s_hues][:extra_vars.n_s_hues]
        #sp.linspace(0.0, 2*sp.pi, n_s_hues, endpoint=False)
    test_sat_dklS = 0.2
    surr_sat_dklS = 0.16
    isolum_el = 0.0 # elevation is 0 to get isolumination to background
    
    stims_all_lms = sp.zeros((extra_vars.n_train, extra_vars.size, extra_vars.size, 3))
    stims_ind_lms = sp.zeros((extra_vars.n_t_hues, extra_vars.n_s_hues, extra_vars.size, extra_vars.size, 3))

    pbar_counter = 0
    pbar = pb((extra_vars.n_s_hues+1)*extra_vars.n_train, 'Building isoluminant stimuli [all]')
    for i, azt in enumerate(all_hues_dkls):
        dklS_ = sp.array([test_sat_dklS, azt, isolum_el])
        c_lms_ = dklC2lms(sph2cart(dklS_))
        stims_all_lms[i, ..., 0] = stim.get_center_surround(
            size=extra_vars.size, csize=extra_vars.csize, cval=c_lms_[0], sval=gray_lms[0])
        stims_all_lms[i, ..., 1] = stim.get_center_surround(
            size=extra_vars.size, csize=extra_vars.csize, cval=c_lms_[1], sval=gray_lms[1])
        stims_all_lms[i, ..., 2] = stim.get_center_surround(
            size=extra_vars.size, csize=extra_vars.csize, cval=c_lms_[2], sval=gray_lms[2])

        pbar_counter += 1
        pbar.update(pbar_counter)
    pbar.finish()

    pbar_counter = 0
    pbar = pb((extra_vars.n_s_hues+1)*extra_vars.n_t_hues, 'Building isoluminant stimuli [ind]')
    for i, azt in enumerate(test_hues_dklS):
        dklS_ = sp.array([test_sat_dklS, azt, isolum_el])
        c_lms_ = dklC2lms(sph2cart(dklS_))
        for j, azs in enumerate(surr_hues_dklS):
            dklS_ = sp.array([surr_sat_dklS, azs, isolum_el])
            s_lms_ = dklC2lms(sph2cart(dklS_))
            stims_ind_lms[i, j, ..., 0] = stim.get_center_surround(
                size=extra_vars.size, csize=extra_vars.csize, cval=c_lms_[0], sval=s_lms_[0])
            stims_ind_lms[i, j, ..., 1] = stim.get_center_surround(
                size=extra_vars.size, csize=extra_vars.csize, cval=c_lms_[1], sval=s_lms_[1])
            stims_ind_lms[i, j, ..., 2] = stim.get_center_surround(
                size=extra_vars.size, csize=extra_vars.csize, cval=c_lms_[2], sval=s_lms_[2])

            pbar_counter += 1
            pbar.update(pbar_counter)
    pbar.finish()

    # compute vanilla SO features for those stimuli
    ###############################################
    so_all = sp.zeros((extra_vars.n_train, nc, extra_vars.size, extra_vars.size))
    so_ind = sp.zeros((extra_vars.n_t_hues, extra_vars.n_s_hues, nc, extra_vars.size, extra_vars.size))

    pbar = pb(extra_vars.n_train, 'Computing SO features [all]')
    for idx in range(extra_vars.n_train):
        so_all[idx] = GET_SO(stims_all_lms[idx],extra_vars._DEFAULT_FLOATX_NP,extra_vars._DEFAULT_KW2015_SO_PARAMETERS)
        pbar.update(idx)
    pbar.finish()

    pbar = pb(extra_vars.n_t_hues*extra_vars.n_s_hues, 'Computing SO features [ind]')
    for idx in range(extra_vars.n_t_hues):
        for jdx in range(extra_vars.n_s_hues):
            so_ind[idx, jdx] = GET_SO(stims_ind_lms[idx, jdx],extra_vars._DEFAULT_FLOATX_NP,extra_vars._DEFAULT_KW2015_SO_PARAMETERS)
            pbar.update(jdx + idx * extra_vars.n_s_hues)
    pbar.finish()
    so_ind = so_ind.reshape(extra_vars.n_t_hues*extra_vars.n_s_hues, 
        nc, extra_vars.size, extra_vars.size)

    #Final ops
    cs_hue_diff = da2ha(test_hues_dklS.reshape(extra_vars.n_t_hues, 1) - \
        surr_hues_dklS.reshape(1, extra_vars.n_s_hues))
    cs_hue_diff *= (180 / sp.pi)
    np.savez(extra_vars.f4_stimuli_file,so_all=so_all,so_ind=so_ind,
        stims_all_lms=stims_all_lms,cs_hue_diff=cs_hue_diff)

Y_reference = 0.75
gray_xyY = sp.array([0.310, 0.316, Y_reference])
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
