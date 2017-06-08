from __future__ import absolute_import
try:
    import colored_traceback.always
except ImportError:
    pass
import h5py,sys
sys.path.append('../../../../')
sys.path.append('../../')
import time
import itertools as it
import argparse
import os
import joblib
from copy import deepcopy
import numpy as np
from numpy import r_, s_
import scipy as sp
from scipy import stats
from scipy.stats import bernoulli
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.optimize import curve_fit, leastsq
from hmax.models.hnorm import models as mod
from hmax.models.hnorm.barams import floatX
from hmax.tools.utils import pb, mul, ifloor, iceil, iround
from hmax.models.hnorm import visualize as viz
from ucircuits.contextual import utils, cutils, contextual_CPU
from ucircuits.contextual import stimuli as stim
from ucircuits.contextual.contextual_CUDA import ContextualCircuit, _DEFAULT_PARAMETERS, _DEFAULT_MAXITER
from ucircuits.contextual.color import experiment_constancy as xxc
import sklearn as sl
from sklearn import linear_model
from sklearn import preprocessing
from skimage.color import xyz2rgb, rgb2xyz
CIECAM02_CAT = sp.array([[ 0.7328,  0.4296, -0.1624],
                         [-0.7036,  1.6975,  0.0061],
                         [ 0.003 ,  0.0136,  0.9834]])

from general_settings import *

_I_GPU = 1
_USE_GPU = True
_N_POINTS = 64
_DECODER_TYPE = None#'circular_vote'
from timeit import default_timer as timer


#------------------------------------------------------------------------------#
_DEFAULT_TB2015PP_CVAL = 0.5
_DEFAULT_TB2015PP_SIZE = 51
_DEFAULT_TB2015PP_CSIZE = 9
_DEFAULT_TB2015PP_SCALE = 2.0 # works with 1.
_DEFAULT_TB2015PP_NPOINTS = 32
_DEFAULT_TB2015PP_CSVFILES = [[
    '/home/dmely/src/hmax/models/ucircuits/contextual' + \
    '/data/TB2015_%i_%s.csv' % (i, s) \
    for i in range(-90, 90, 30)] for s in ('PS', 'PO')
]


def TB2015_plaid(
    size=_DEFAULT_TB2015PP_SIZE,
    csize=_DEFAULT_TB2015PP_CSIZE,
    npoints=_DEFAULT_TB2015PP_NPOINTS,
    scale=_DEFAULT_TB2015PP_SCALE,
    cval=_DEFAULT_TB2015PP_CVAL):
    """
    """

    BUFFERFILE = '/home/dmely/data/buffer_tb2015_plaid_cs5.h5'

    # map to a sine for the visualizations
    m = (lambda x: sp.sin(sp.pi * x))

    # simulate populations
    ppop = {
        'kind': 'circular',
        'npoints': npoints,
        'scale': scale,
        'fdomain': (0, 1),
    }

    vals_ang = sp.array([-90., -60., -30., 0., 30., 60.])
    vals = (vals_ang + 90.)/180.
    imc1 = stim.get_center_surround(
        size=size, csize=csize, cval=cval, sval=sp.nan)
    x1 = utils.get_population(imc1, **ppop)
    x = sp.zeros((2, len(vals), npoints, size, size))
    
    for vdx, v in enumerate(vals):
        imc2 = stim.get_center_surround(
            size=size, csize=csize, cval=v, sval=sp.nan)
        ims = stim.get_center_surround(
            size=size, csize=csize, cval=sp.nan, sval=v)
        x2 = utils.get_population(imc2, **ppop)
        xs = utils.get_population(ims, **ppop)
        x[0, vdx] = (x1 + x2)/2.
        x[1, vdx] = (x1 + x2)/2. + xs
    x.shape = (2 * len(vals), npoints, size, size)

    try:
        with h5py.File(BUFFERFILE, 'r') as h5f:
            pr = h5f['pr'][:]

    except IOError:
        with ContextualCircuit(input_shape=x.shape,
            keeptime=False, i_gpu=_I_GPU) as cx:
            cx.run(x, from_gpu=False)
            pr = cx.Y[..., size//2, size//2].get()
            pr.shape = (2, len(vals), npoints)

        with h5py.File(BUFFERFILE) as h5f:
            h5f['pr'] = pr

    # display population responses for plaids and surrounds
    # raw_input("[Populations] Press Enter to continue.")
    xticks = sp.array([-90., -60., -30., 0., 30., 60., 90.])
    allx = sp.linspace(xticks.min(), xticks.max(), npoints)
    
    fig, ax = plt.subplots(2, len(vals))
    ax_dat = ax[0, :]
    ax_sim = ax[1, :]

    # plot digitized data
    _plot_TrottBorn2015_population_plaid_data(ax=ax_dat)

    d0 = utils.decode(pr[0], axis=-1, kind='circular_vote')
    d1 = utils.decode(pr[1], axis=-1, kind='circular_vote')
    d0_x = d0 * (xticks.max()-xticks.min())+xticks.min()
    d1_x = d1 * (xticks.max()-xticks.min())+xticks.min()

    # plot simulated
    for i in range(len(vals)):
        # population curve
        ax_sim[i].plot(allx, pr[0, i],
            color=CELL_LINECOLOR,
            alpha=CELL_LINEALPHA,
            markersize=0,
            label='Plaid only',
            linestyle='-',
            linewidth=CELL_LINEWIDTH)

        ax_sim[i].plot(allx[::iceil(npoints/6.)], pr[0, i][::iceil(npoints/6.)],
            color=CELL_MARKERCOLOR,
            alpha=CELL_MARKERALPHA,
            linestyle='None',
            markersize=CELL_MARKERSIZE,
            marker='o')

        ax_sim[i].plot(allx, pr[1, i],
            color=CELL_LINECOLOR_RED,
            alpha=CELL_LINEALPHA,
            markersize=0,
            label='Plaid + Surround',
            linestyle='-',
            linewidth=CELL_LINEWIDTH)

        ax_sim[i].plot(allx[::iceil(npoints/6.)], pr[1, i][::iceil(npoints/6.)],
            color=CELL_MARKERCOLOR_RED,
            alpha=CELL_MARKERALPHA,
            linestyle='None',
            markersize=CELL_MARKERSIZE,
            marker='o')
        ax_sim[i].set_ylim([0.0, pr.max()])

        # misc
        ax_sim[i].set_xticks(xticks)
        ax_sim[i].set_xticklabels('%i' % (i,) for i in xticks)
        ax_sim[i].set_xlim((xticks.min(), xticks.max()))

        if not PUBLISH:
            if xticks[i] < 0:
                tt = 'C2 = C1 - %i deg.' % (sp.absolute(xticks[i]),)
            elif xticks[i] > 0:
                tt = 'C2 = C1 + %i deg.' % (sp.absolute(xticks[i]),)
            else:
                tt = 'C2 = C1'
            ax_dat[i].set_title(tt, fontweight='bold', fontsize=15.)
            ax_sim[i].set_xlabel('Orientation\n(relative to preferred)',
                fontweight='bold', fontsize=15.)

            if i == 0:
                ax_dat[i].set_ylabel(
                    'Population Response (normalized)',
                    fontweight='bold', fontsize=15.)
                ax_sim[i].set_ylabel(
                    'Population Response (arbitrary units)',
                    fontweight='bold', fontsize=15.)
            if i == len(vals)-1:
                ax_dat[i].yaxis.set_label_position('right')
                ax_sim[i].yaxis.set_label_position('right')
                ax_dat[i].set_ylabel(
                    'Experimental data (neurophysiology)',
                    fontweight='bold', fontsize=15.)
                ax_sim[i].set_ylabel(
                    'Model simulation',
                    fontweight='bold', fontsize=15.)
            # ax_sim[i].patch.set_alpha(0.0)

    for i in range(6):
        if PUBLISH:
            ax_dat[i].set_xticklabels([])
            ax_dat[i].set_yticklabels([])
            ax_sim[i].set_xticklabels([])
            ax_sim[i].set_yticklabels([])

        ax_sim[i].set_xlim((-93, 93))
        ax_dat[i].set_xlim((-93, 93))

        # minima loci
        ax_sim[i].plot([d0_x[i]]*2, list(ax_sim[i].get_ylim()),
            color=CELL_LINECOLOR,
            alpha=CELL_LINEALPHA,
            linewidth=CELL_ABSCISSA_LW,
            linestyle='--')
        ax_sim[i].plot([d1_x[i]]*2, list(ax_sim[i].get_ylim()),
            color=CELL_LINECOLOR_RED,
            alpha=CELL_LINEALPHA,
            linewidth=CELL_ABSCISSA_LW,
            linestyle='--')        

    if PUBLISH:
        CELLFIG(fig=fig, out='/home/dmely/data/cell.out/fig_tb15_pl.pdf')

    return pr