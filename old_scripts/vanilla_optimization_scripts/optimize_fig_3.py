from __future__ import absolute_import
import sys,os,argparse
sys.path.append('/home/drew/Documents/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from copy import deepcopy
import numpy as np
import scipy as sp
from hmax.models.ucircuits.contextual import stimuli as stim
from parameter_defaults import PaperDefaults
from ops.hp_optim import *

def optim_f3(lesions,out_dir):
    defaults = PaperDefaults()
    if lesions != None:
        defaults.lesions = lesions
    #David's globals
    _DECODER_TYPE = None#'circular_vote'
    _DEFAULT_TILTEFFECT_DEGPERPIX = .25 # <OToole77>
    _DEFAULT_TILTEFFECT_SIZE = 51#101
    _DEFAULT_TILTEFFECT_CSIZE = iround(2. / _DEFAULT_TILTEFFECT_DEGPERPIX)
    _DEFAULT_TILTEFFECT_SSIZE = iround(8. / _DEFAULT_TILTEFFECT_DEGPERPIX)
    _DEFAULT_TILTEFFECT_CVAL = .5
    _DEFAULT_TILTEFFECT_SVALS = np.linspace(0.0, 0.5, 10)
    _DEFAULT_TILTEFFECT_SCALES = {'ow77': 0.40, 'ms79': 0.60}#0.45
    _DEFAULT_TILTEFFECT_NPOINTS = 25#100
    _DEFAULT_TILTEFFECT_CIRCULAR = True
    _DEFAULT_TILTEFFECT_DECODER_TYPE = 'circular_vote'
    _DEFAULT_TILTEFFECT_CSV = {
        'ow77': os.path.join(defaults._DATADIR, 'OW_fig4_Black.csv'),
        'ms79': os.path.join(defaults._DATADIR, 'MS1979.csv'),
    }

    # experiment parameters
    m = (lambda x: sp.sin(sp.pi * x)) if _DEFAULT_TILTEFFECT_CIRCULAR else (lambda x: x)
    assert len(_DEFAULT_TILTEFFECT_CSV) == 2
    assert len(_DEFAULT_TILTEFFECT_SCALES) == 2
    cpt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2)
    spt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2 + _DEFAULT_TILTEFFECT_CSIZE)
    dt_in = _DEFAULT_TILTEFFECT_CVAL - _DEFAULT_TILTEFFECT_SVALS
    dtmin = dt_in.min()
    dtmax = dt_in.max()

    # simulate populations
    im = sp.array([[stim.get_center_nfsurrounds(size=_DEFAULT_TILTEFFECT_SIZE,
        csize=_DEFAULT_TILTEFFECT_CSIZE, nsize=_DEFAULT_TILTEFFECT_CSIZE, fsize=_DEFAULT_TILTEFFECT_SSIZE,
        cval=_DEFAULT_TILTEFFECT_CVAL, nval=_DEFAULT_TILTEFFECT_CVAL, fval=sval,
        bgval=sp.nan)] for sval in _DEFAULT_TILTEFFECT_SVALS])

    # get shifts for model for both papers, and from digitized data
    sortidx = sp.argsort(dt_in) # re-order in increasing angular differences

    # O'Toole and Wenderoth (1977)
    _, ds_ow77_paper_y = sp.genfromtxt(_DEFAULT_TILTEFFECT_CSV['ow77'], delimiter=',').T

    extra_vars = {}
    extra_vars['scale'] = _DEFAULT_TILTEFFECT_SCALES['ow77']
    extra_vars['decoder'] = _DEFAULT_TILTEFFECT_DECODER_TYPE
    extra_vars['npoints'] = _DEFAULT_TILTEFFECT_NPOINTS
    extra_vars['npoints'] = _DEFAULT_TILTEFFECT_NPOINTS
    extra_vars['cval'] = _DEFAULT_TILTEFFECT_CVAL
    extra_vars['sortidx'] = sortidx
    extra_vars['cpt'] = cpt
    extra_vars['spt'] = spt
    extra_vars['sval'] = sval
    extra_vars['kind'] = 'circular'
    extra_vars['figure_name'] = 'f3'
    extra_vars['return_var'] = 'O'

    scores,params = optimize_model(im,ds_ow77_paper_y,extra_vars,defaults)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez(os.path.join(out_dir,'hp_optimization_lesion_' + '_'.join(defaults.lesions)),scores=scores,params=params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesions", dest="lesions",nargs='*',
        default=None, help="Where to lesion the model. Enter lesions as space seperated strings. Use for running batches of hp optims.")
    parser.add_argument("--out_dir", type=str, dest="out_dir",
        default="optim_npys", help="Output directory for hyperparemter optimization npys.")
    args = parser.parse_args()
    optim_f3(**vars(args))
