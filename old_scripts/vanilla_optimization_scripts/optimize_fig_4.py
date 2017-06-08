from __future__ import absolute_import
import sys,os,argparse
sys.path.append('/home/drew/Documents/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from copy import deepcopy
import numpy as np
import scipy as sp
from parameter_defaults import PaperDefaults
from ops.hp_optim import *
from ops.fig_4_utils import *
from ops.model_utils import sfit
from scipy.interpolate import UnivariateSpline

def optim_f4(lesions,out_dir,initialize_model=False):
    defaults = PaperDefaults()
    if lesions != None:
        defaults.lesions = lesions

    #David's globals
    _DEFAULT_KW2015_SO_PARAMETERS = {
        'filters': {'name': 'gabors', 'aspect_ratio': .6,
        'sizes': sp.array([9]), 'spatial_frequencies': sp.array([[9.0]]),
        'orientations': sp.arange(2)*sp.pi/2, 'phases': sp.array([0]),
        'with_center_surround': False, 'padding': 'reflect',
        'corr': False, 'ndp': False},
        'model': {'channels_so': ('R+G-', 'B+Y-', 'R+C-', 'Wh+Bl-',
            'G+R-', 'Y+B-', 'C+R-', 'Bl+Wh-'),'normalize': False},
        'dnp_so': None, 'selected_channels': [0, 1, 3, 4, 5, 7],
        'norm_channels': [0, 1, 3, 4, 5, 7]}

    size=51
    csize=9
    n_train=32
    n_t_hues=16
    n_s_hues=16
    cxp=None
    csvfiles=[
        defaults._DATADIR + '/KW2015_%i.csv' % (i,) for i in range(0, 360, 45)
    ]
    percent_reg_train=80.

    #Load data from experiments
    kw2015_fig2_x = sp.zeros((len(csvfiles), 16))
    kw2015_fig2_y = sp.zeros((len(csvfiles), 16))
    for idx, csv in enumerate(csvfiles):
        kw2015_fig2_x[idx], kw2015_fig2_y[idx] = \
            sp.genfromtxt(csv, delimiter=',')[1:].T
    spl = UnivariateSpline(
        x=kw2015_fig2_x.mean(0),
        y=kw2015_fig2_y.mean(0))
    kw2015_fig2_x_fit = sp.linspace(
            kw2015_fig2_x.mean(0).min(),
            kw2015_fig2_x.mean(0).max(), 360)
    kw2015_fig2_y_fit = spl(kw2015_fig2_x_fit)

    # experiment stimuli
    extra_vars = {}
    extra_vars['_DEFAULT_KW2015_SO_PARAMETERS'] = _DEFAULT_KW2015_SO_PARAMETERS
    extra_vars['_DEFAULT_FLOATX_NP'] = defaults._DEFAULT_FLOATX_NP
    extra_vars['size'] = size
    extra_vars['csize'] = csize
    extra_vars['n_train'] = n_train
    extra_vars['n_t_hues'] = n_t_hues
    extra_vars['n_s_hues'] = n_s_hues
    extra_vars['figure_name'] = 'f4'
    extra_vars['gt_x'] = kw2015_fig2_x_fit
    extra_vars['f4_stimuli_file'] = defaults.f4_stimuli_file
    extra_vars['return_var'] = 'I'
    extra_vars['precalculated_x'] = True

    if initialize_model:
        create_stims(extra_vars)
    stim_files = np.load(extra_vars['f4_stimuli_file'])

    #Run model
    #cx.run(so_all, from_gpu=False)
    #sx_all[:] = cx.Y.get()[:, :, size//2, size//2]
    scores,params = optimize_model(
        stim_files['so_ind'].
        reshape(n_t_hues*n_s_hues,len(_DEFAULT_KW2015_SO_PARAMETERS['norm_channels']),size,size).
        transpose(0,2,3,1),
        np.mean(kw2015_fig2_y,axis=0),extra_vars,defaults)

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
    optim_f4(**vars(args))
