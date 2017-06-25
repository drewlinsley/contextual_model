from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import scipy as sp
from ops import stimuli as stim
from ops.model_utils import iround
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_db_hp_optim import optimize_model


def run():
    defaults = PaperDefaults()

    #David's globals
    _DEFAULT_TILTEFFECT_DEGPERPIX = .25 # <OToole77>
    _DEFAULT_TILTEFFECT_SIZE = 51#101
    _DEFAULT_TILTEFFECT_CSIZE = iround(2. / _DEFAULT_TILTEFFECT_DEGPERPIX)
    _DEFAULT_TILTEFFECT_SSIZE = iround(8. / _DEFAULT_TILTEFFECT_DEGPERPIX)
    _DEFAULT_TILTEFFECT_CVAL = .5
    _DEFAULT_TILTEFFECT_SVALS = np.linspace(0.0, 0.5, 10)
    _DEFAULT_TILTEFFECT_SCALES = {'ow77': 0.40, 'ms79': 0.60}#0.45
    _DEFAULT_TILTEFFECT_NPOINTS = 25#100
    _DEFAULT_TILTEFFECT_DECODER_TYPE = 'circular_vote'
    _DEFAULT_TILTEFFECT_CSV = {
        'ow77': os.path.join(defaults._DATADIR, 'OW_fig4_Black.csv'),
        'ms79': os.path.join(defaults._DATADIR, 'MS1979.csv'),
    }

    # experiment parameters
    cpt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2)
    spt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2 + _DEFAULT_TILTEFFECT_CSIZE)
    dt_in = _DEFAULT_TILTEFFECT_CVAL - _DEFAULT_TILTEFFECT_SVALS

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
    extra_vars['figure_name'] = 'f3a'
    extra_vars['return_var'] = 'O'
    optimize_model(im, ds_ow77_paper_y, extra_vars, defaults)

if __name__ == '__main__':
    run()
