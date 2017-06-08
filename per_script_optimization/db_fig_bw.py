from __future__ import absolute_import
import sys,os
sys.path.append('/home/drew/Documents/')
sys.path.append('/home/drew/Documents/tf_experiments/experiments/contextual_circuit/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import scipy as sp
from hmax.models.ucircuits.contextual import stimuli as stim
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_db_hp_optim import optimize_model
from ops import model_utils

def run(hps=None):
    defaults = PaperDefaults()

    #David's globals
    size=51
    csize=9
    npoints=37
    scale=1.
    _DEFAULT_BWC_CSV_CTS = sp.array([0.0, .06, .12, .25, .50]) * 100
    csvfiles=sp.array([[os.path.join(defaults._DATADIR, 'BWC2009_%i_%i.csv' \
    % (i, j)) for i in _DEFAULT_BWC_CSV_CTS] for j in _DEFAULT_BWC_CSV_CTS]).T

    # experiment parameters
    im = sp.array([
        stim.get_center_surround(
            size=size, csize=csize, cval=.25, sval=sp.nan),
        stim.get_center_surround(
            size=size, csize=csize, cval=.75, sval=sp.nan)])

    # populations for vertical (masking) and horizontal (driving) stimuli
    #####################################################################
    xv = model_utils.get_population(im[0],
        kind='circular', npoints=npoints, scale=scale)
    xh = model_utils.get_population(im[1],
        kind='circular', npoints=npoints, scale=scale)

    # superimposed populations
    ##########################
    v_contrasts = [0.0, .06, .12, .25, .50]
    h_contrasts = [0.0, .06, .12, .25, .50]
    nv, nh = len(v_contrasts), len(h_contrasts)
    x = sp.array([[h*xh + v*xv for h in h_contrasts] for v in v_contrasts])
    x.shape = (nv * nh,) + x.shape[2:]

    # busse and wade data
    t_paper = sp.zeros((nv, nh, 13))
    y_paper = sp.zeros((nv, nh, 13))

    for idx in range(nv):
        for jdx in range(nh):
            t_paper[idx, jdx], y_paper[idx, jdx] = \
                sp.genfromtxt(csvfiles[idx, jdx], delimiter=',').T

    res_y_paper = sp.zeros((y_paper.shape[0],y_paper.shape[1],npoints))
    for r in range(y_paper.shape[0]):
        for c in range(y_paper.shape[1]):
            res_y_paper[r,c,:] = sp.signal.resample(y_paper[r,c,:],npoints)
    gt = [t_paper,res_y_paper]            

    extra_vars = {}
    extra_vars['scale'] = scale
    extra_vars['npoints'] = npoints
    extra_vars['size'] = size
    extra_vars['csize'] = csize
    extra_vars['nv'] = nv
    extra_vars['nh'] = nh
    extra_vars['figure_name'] = 'bw'
    extra_vars['return_var'] = 'O'

    optimize_model(x,gt,extra_vars,defaults)

if __name__ == '__main__':
    run()
