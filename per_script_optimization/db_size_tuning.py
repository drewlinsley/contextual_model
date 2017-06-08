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
import seaborn as sns 

def run():
    defaults = PaperDefaults()

    #David's globals
    size=51
    nr=17
    npoints=32//2
    ncontrasts=5

    # experiment parameters
    # generate stimuli
    ##################
    im = []
    for k in range(nr):
        im_ = sp.zeros((size, size)) + sp.nan
        im_[size//2-k:size//2+k+1, size//2-k:size//2+k+1] = 0.5
        im.append(im_)
    im = sp.array(im)

    # generate populations
    ######################
    contrasts = sp.linspace(1., 0., ncontrasts, endpoint=False)[::-1]
    # contrasts = sp.logspace(-2, 0., ncontrasts)
    x = sp.array([utils.get_population(
        im_, 'gaussian', npoints=npoints) for im_ in im])
    ax = [c*x for c in contrasts]
    cx = np.concatenate(ax[:],axis=0)
    
    # Experimental data
    extra_vars = {}
    extra_vars['size'] = size
    extra_vars['npoints'] = npoints 
    extra_vars['nr'] = nr
    extra_vars['stimsizes'] = 2*sp.arange(nr)+1
    extra_vars['ssn'] = defaults._DEFAULT_PARAMETERS['ssn']
    extra_vars['ssf'] = defaults._DEFAULT_PARAMETERS['ssf']
    extra_vars['hp_file'] = 'best_hps.npz'
    extra_vars['figure_name'] = 'size_tuning'
    extra_vars['return_var'] = 'O'
    extra_vars['contrasts'] = contrasts
    extra_vars['curvecols'] = sns.cubehelix_palette(ncontrasts)
    extra_vars['curvelabs'] = ['Single-cell response at contrast %g' \
            % (cst,) for cst in contrasts]
 
    optimize_model(cx,extra_vars,defaults)

if __name__ == '__main__':
    run()
