from __future__ import absolute_import
try:
    import colored_traceback.always
except ImportError:
    pass
import h5py,sys,os
sys.path.append('/home/drew/Documents/')
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
from hmax.models.ucircuits.contextual import stimuli as stim
from hmax.models.ucircuits.contextual.contextual_CUDA import _DEFAULT_PARAMETERS, _DEFAULT_MAXITER
from hmax.models.ucircuits.contextual.color import experiment_constancy as xxc
import sklearn as sl
from sklearn import linear_model
from sklearn import preprocessing
import tensorflow as tf
from utils import utils
from utils.cell_figure_settings import *
from parameter_defaults import PaperDefaults

from tensorflow.python.client import timeline
#from model_defs.model_cpu_port_nchw import ContextualCircuit
from model_defs.model_cpu_port_scan import ContextualCircuit
from timeit import default_timer as timer

_N_POINTS = 64
_DECODER_TYPE = None#'circular_vote'
profile = False
defaults = PaperDefaults()

# helper functions
##################
def compute_shifts(stimuli, scale):
    """ Get shifts for given tuning bandwidth """
    x = utils.get_population2(stimuli,
        npoints=npoints, kind='circular', scale=scale).astype(defaults._DEFAULT_FLOATX_NP).transpose(0,2,3,1) #transpose to bhwc
    start = timer()
    with tf.device('/gpu:0'):
        ctx = ContextualCircuit()
        ctx.run(x) #builds tf graph with shape of x
    end = timer()
    print(end - start)      

    with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
        start = timer()
        if profile:
            #chrome://tracing
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            y = sess.run(ctx.out_O,feed_dict={ctx.X:x},options=run_options,run_metadata=run_metadata)

            #Profiling        
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
        else:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            y = sess.run(ctx.out_O,feed_dict={ctx.X:x})
        end = timer()
        print(end - start)      

    #transpose x and y back to bchw
    x = x.transpose(0,3,1,2)
    y = y.transpose(0,3,1,2)

    # compute absolute and relative dshifts
    xdec = utils.decode(x[:, :, cpt[0], cpt[1]], axis=1, kind=decoder_type)
    cdec = utils.decode(y[:, :, cpt[0], cpt[1]], axis=1, kind=decoder_type)
    sdec = utils.decode(y[:, :, spt[0], spt[1]], axis=1, kind=decoder_type)
    dshift_a = cdec - xdec
    dshift_r = (cdec - sdec)/(cval - sval)
    return dshift_a, dshift_r

def sfit(ptx_eval, ptx, pty, s=1., k=3, t=None):
    """ Fit a bunch of points and return an evaluation vector """
    assert len(ptx) == len(pty)
    if t is None:
        return UnivariateSpline(x=ptx, y=pty, s=s, k=k)(ptx_eval)
    else:
        return LSQUnivariateSpline(x=ptx, y=pty, t=t, k=k)(ptx_eval)

""" Better with center size = 7 and gamma = .25 """
#Figure 3c
_DEFAULT_TILTEFFECT_DEGPERPIX = .25 # <OToole77>
_DEFAULT_TILTEFFECT_SIZE = 51#101
_DEFAULT_TILTEFFECT_CSIZE = iround(2. / _DEFAULT_TILTEFFECT_DEGPERPIX)
_DEFAULT_TILTEFFECT_SSIZE = iround(8. / _DEFAULT_TILTEFFECT_DEGPERPIX)
_DEFAULT_TILTEFFECT_CVAL = .5
_DEFAULT_TILTEFFECT_SVALS = sp.linspace(0.0, 0.5, 10)
_DEFAULT_TILTEFFECT_SCALES = {'ow77': 0.40, 'ms79': 0.60}#0.45
_DEFAULT_TILTEFFECT_NPOINTS = 25#100
_DEFAULT_TILTEFFECT_CIRCULAR = True
_DEFAULT_TILTEFFECT_DECODER_TYPE = 'circular_vote'
_DEFAULT_TILTEFFECT_CSV = {
    'ow77': os.path.join(DATADIR, 'OW_fig4_Black.csv'),
    'ms79': os.path.join(DATADIR, 'MS1979.csv'),
}

_DEFAULT_VALUE_BW3DS_SIZE = 31
_DEFAULT_VALUE_BW3DS_CSIZE = 8
_DEFAULT_VALUE_BW3DS_CVAL = .5
_DEFAULT_VALUE_BW3DS_NSCALES = 49#49
_DEFAULT_VALUE_BW3DS_SVALS = sp.linspace(0.0, 0.5, 11)
_DEFAULT_VALUE_BW3DS_BWS = sp.linspace(1., 75., 50)*sp.pi/90.#sp.linspace(0.20, 2., 50)
_DEFAULT_VALUE_BW3DS_CIRCULAR = True
_DEFAULT_VALUE_BW3DS_DECODER_TYPE = 'circular_vote'


#Assign variables
size=_DEFAULT_TILTEFFECT_SIZE
csize=_DEFAULT_TILTEFFECT_CSIZE
ssize=_DEFAULT_TILTEFFECT_SSIZE
cval=_DEFAULT_TILTEFFECT_CVAL
svals=_DEFAULT_TILTEFFECT_SVALS
scales=_DEFAULT_TILTEFFECT_SCALES
npoints=_DEFAULT_TILTEFFECT_NPOINTS
circular=_DEFAULT_TILTEFFECT_CIRCULAR
decoder_type=_DEFAULT_TILTEFFECT_DECODER_TYPE
csvfiles=_DEFAULT_TILTEFFECT_CSV

# map to a sine
m = (lambda x: sp.sin(sp.pi * x)) if circular else (lambda x: x)

# experiment parameters
#######################

assert len(csvfiles) == 2
assert len(scales) == 2
nsize=csize
fsize=ssize
cpt = (size//2, size//2)
spt = (size//2, size//2 + csize)
dt_in = cval - svals
dtmin = dt_in.min()
dtmax = dt_in.max()

# simulate populations
######################
im = sp.array([[stim.get_center_nfsurrounds(size=size,
    csize=csize, nsize=csize, fsize=ssize,
    cval=cval, nval=cval, fval=sval,
    bgval=sp.nan)] for sval in svals])

# get shifts for model for both papers, and from digitized data
###############################################################

# O'Toole and Wenderoth (1977)
ds_ow77_abs, ds_ow77_rel = compute_shifts(im, scale=scales['ow77'])
ds_ow77_paper_x, ds_ow77_paper_y = \
    sp.genfromtxt(csvfiles['ow77'], delimiter=',').T

# with h5py.File('/home/dmely/data/ow77ms79_buffer.h5') as hdf:
#     ds_ow77_abs = hdf['ds_ow77_abs'][:]
#     ds_ow77_rel = hdf['ds_ow77_rel'][:]
#     ds_ms79_abs = hdf['ds_ms79_abs'][:]
#     ds_ms79_rel = hdf['ds_ms79_rel'][:]

# re-order in increasing angular differences
sortidx = sp.argsort(dt_in)
dt_in = dt_in[sortidx]
ds_ow77_abs = ds_ow77_abs[sortidx]
ds_ow77_rel = ds_ow77_rel[sortidx]

# convert back from [0, 1] range to degrees
ds_ow77_abs *= 180.

# fit model and paper data
##########################
dt_sim = sp.linspace(dtmin, dtmax, 100)
dt_180 = dt_sim * 360.
dt_90 = dt_sim * 180.

ds_ow77_paper_fit = sfit(dt_90,
    ptx=ds_ow77_paper_x, pty=ds_ow77_paper_y, t=[45.])
ds_ow77_model_fit = sfit(dt_sim,
    ptx=dt_in, pty=ds_ow77_abs, t=[30./180.])

# plots
#######
fig, ax = plt.subplots(1, 2)

# O'Toole & Wenderoth (1977): original data
ax[0].plot(ds_ow77_paper_x, ds_ow77_paper_y,
    linestyle='None', marker='o',
    markersize=CELL_MARKERSIZE,
    color=CELL_MARKERCOLOR,
    alpha=CELL_MARKERALPHA)
ax[0].plot(dt_90, ds_ow77_paper_fit,
    linewidth=CELL_LINEWIDTH,
    linestyle='-', markersize=0,
    color=CELL_LINECOLOR,
    alpha=CELL_LINEALPHA)

# O'Toole & Wenderoth (1977): simulation data
ax[1].plot(dt_in, ds_ow77_abs,
    linestyle='None', marker='o',
    markersize=CELL_MARKERSIZE,
    color=CELL_MARKERCOLOR,
    alpha=CELL_MARKERALPHA)
ax[1].plot(dt_sim, ds_ow77_model_fit,
    linewidth=CELL_LINEWIDTH,
    linestyle='-', markersize=0,
    color=CELL_LINECOLOR,
    alpha=CELL_LINEALPHA)

for jdx in range(2):
    ax[jdx].plot( # zero-ordinate line
        [min(ax[jdx].get_xlim()), max(ax[jdx].get_xlim())],
        [0.0, 0.0],
        color=CELL_ABSCISSA_C,
        linewidth=CELL_ABSCISSA_LW,
        alpha=CELL_ABSCISSA_ALPHA)

ax[1].yaxis.set_label_position('right')
ax[1].yaxis.set_ticks_position('right')

# x-axes limits, ticks, etc.
nxticks = 4
ax[0].set_xlim([0.0, 90.0])
ax[0].set_xticks(sp.linspace(0, 90, nxticks))
ax[0].set_xticklabels(sp.linspace(0, 90, nxticks, dtype=int))

ax[1].set_xlim([0.0, 0.5])
ax[1].set_xticks(sp.linspace(0, 0.5, nxticks))
ax[1].set_xticklabels(sp.linspace(0, 90, nxticks, dtype=int))

# retick(ax[1], n=len(ax[0].get_yticks()), k=2)
ax[0].set_ylim([-1.0, 3.5])
ax[1].set_ylim([-1.0, 3.5])

out_dir = 'figures'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
CELLFIG(fig=fig, out=os.path.join(out_dir,'fig3a.pdf'))
