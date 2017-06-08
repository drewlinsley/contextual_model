from __future__ import absolute_import
import sys,os
sys.path.append('/home/drew/Documents/')
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import scipy as sp
from hmax.models.ucircuits.contextual import stimuli as stim
from hmax.tools.utils import iround
from ops.parameter_defaults import PaperDefaults
from ops.single_hp_optim import optimize_model
from ops import model_utils as utils

def run():
    defaults = PaperDefaults()

    #David's globals
    size=51
    npoints=64
    cval1=0.25
    cval2=0.75
    sval=0.75
    test_contrasts=sp.array([0., 8., 32.])
    mask_contrasts=sp.array([0., 8., 32.])

    # experiment parameters
    ntests, nmasks = len(test_contrasts), len(mask_contrasts)
    decoder_type = 'circular_vote'
    idx1 = int(cval1 * npoints)
    idx2 = int(cval2 * npoints)

    # simulate populations
    imc = stim.get_center_surround(
        size=size, csize=9, cval=cval1, sval=sp.nan)
    ims = stim.get_center_surround(
        size=size, csize=9, cval=sp.nan, sval=sval)
    x1 = utils.get_population(imc,
        npoints=npoints,
        kind='gaussian',
        scale=0.1,
        fdomain=(0, 1))
    x2 = sp.roll(x1, int((cval2 - cval1) * npoints), axis=-3)
    xs = utils.get_population(ims,
        npoints=npoints,
        kind='gaussian',
        scale=0.1,
        fdomain=(0, 1))
    x = []

    for k1 in test_contrasts:
        for k2 in mask_contrasts:
            x.append(k1/100. * x1 + k2/100. * x2)
    x = sp.array(x) + sp.array([xs])

    # Experimental data
    extra_vars = {}
    extra_vars['size'] = size
    extra_vars['npoints'] = npoints 
    extra_vars['sval'] = sval
    extra_vars['figure_name'] = 'cross_orientation_suppression'
    extra_vars['return_var'] = 'O'
    extra_vars['idx1'] = idx1
    extra_vars['idx2'] = idx2
    extra_vars['test_contrasts'] = test_contrasts
    extra_vars['mask_contrasts'] = mask_contrasts
    optimize_model(x,[],extra_vars,defaults)

if __name__ == '__main__':
    run()
