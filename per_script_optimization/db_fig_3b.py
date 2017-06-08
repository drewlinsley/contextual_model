from __future__ import absolute_import
import sys,os
sys.path.append('/home/drew/Documents/')
sys.path.append('/home/drew/Documents/tf_experiments/experiments/contextual_circuit/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import scipy as sp
from scipy import signal 
from hmax.models.ucircuits.contextual import stimuli as stim
from hmax.tools.utils import iround
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_db_hp_optim import optimize_model

def run():
    defaults = PaperDefaults()

    #David's globals
    _DEFAULT_KW97_TILTEFFECT_DEGPERPIX = .45 # <OToole77>
    _DEFAULT_TILTEFFECT_SIZE = 101#101
    _DEFAULT_KW97_TILTEFFECT_CSIZE = iround(3.6 / _DEFAULT_KW97_TILTEFFECT_DEGPERPIX)
    _DEFAULT_KW97_TILTEFFECT_NSIZE = iround(5.4 / _DEFAULT_KW97_TILTEFFECT_DEGPERPIX)
    _DEFAULT_KW97_TILTEFFECT_FSIZE = iround(10.7 / _DEFAULT_KW97_TILTEFFECT_DEGPERPIX)
    _DEFAULT_TILTEFFECT_CVAL = .5
    _DEFAULT_TILTEFFECT_SVALS = np.linspace(0.0, 0.5, 10)
    _DEFAULT_KW97_TILTEFFECT_SCALE = 1.25
    _DEFAULT_TILTEFFECT_NPOINTS = 25#100
    _DEFAULT_TILTEFFECT_CIRCULAR = True
    _DEFAULT_TILTEFFECT_DECODER_TYPE = 'circular_vote'
    csvfiles = [
        os.path.join(defaults._DATADIR, 'KW97_GH.csv'),
        os.path.join(defaults._DATADIR, 'KW97_JHK.csv'),
        os.path.join(defaults._DATADIR, 'KW97_LL.csv'),
        os.path.join(defaults._DATADIR, 'KW97_SJL.csv'),
    ]

    # experiment parameters
    cpt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2)
    spt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2 + _DEFAULT_KW97_TILTEFFECT_CSIZE)
    dt_in = _DEFAULT_TILTEFFECT_CVAL - _DEFAULT_TILTEFFECT_SVALS

    # simulate populations
    im = sp.array([[stim.get_center_nfsurrounds(size=_DEFAULT_TILTEFFECT_SIZE,
        csize=_DEFAULT_KW97_TILTEFFECT_CSIZE, nsize=_DEFAULT_KW97_TILTEFFECT_NSIZE, fsize=_DEFAULT_KW97_TILTEFFECT_FSIZE,
        cval=_DEFAULT_TILTEFFECT_CVAL, nval=sp.nan, fval=sval,
        bgval=sp.nan)] for sval in _DEFAULT_TILTEFFECT_SVALS])

    # get shifts for model for both papers, and from digitized data
    sortidx = sp.argsort(dt_in) # re-order in increasing angular differences

    # O'Toole and Wenderoth (1977)
    n_subjects = len(csvfiles)
    ds_kw97_paper_x = sp.zeros((n_subjects, 9))
    ds_kw97_paper_y = sp.zeros((n_subjects, 9))

    for sidx, csv in enumerate(csvfiles):
        ds_kw97_paper_x[sidx], ds_kw97_paper_y[sidx] = \
            sp.genfromtxt(csv, delimiter=',').T

    ds_kw97_paper_x = (ds_kw97_paper_x + 360.) % 360. - 45.
    ds_kw97_paper_y = 45. - ds_kw97_paper_y

    for sidx in range(n_subjects):
        ds_kw97_paper_x[sidx] = ds_kw97_paper_x[sidx][
            sp.argsort(ds_kw97_paper_x[sidx])]

    extra_vars = {}
    extra_vars['scale'] = _DEFAULT_KW97_TILTEFFECT_SCALE
    extra_vars['decoder'] = _DEFAULT_TILTEFFECT_DECODER_TYPE
    extra_vars['npoints'] = _DEFAULT_TILTEFFECT_NPOINTS
    extra_vars['npoints'] = _DEFAULT_TILTEFFECT_NPOINTS
    extra_vars['cval'] = _DEFAULT_TILTEFFECT_CVAL
    extra_vars['sortidx'] = sortidx
    extra_vars['cpt'] = cpt
    extra_vars['spt'] = spt
    extra_vars['sval'] = sval
    extra_vars['kind'] = 'circular'
    extra_vars['figure_name'] = 'f3b'
    extra_vars['return_var'] = 'O'

    adjusted_gt = signal.resample(np.mean(ds_kw97_paper_y,axis=0),10)
    optimize_model(im,adjusted_gt,extra_vars,defaults)

if __name__ == '__main__':
    run()
