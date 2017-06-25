from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import scipy as sp
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_db_hp_optim import optimize_model
from ops.fig_4_utils import *
from ops.model_utils import sfit


def run(initialize_model=False):
    defaults = PaperDefaults()

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
    csvfiles=[
        defaults._DATADIR + '/KW2015_%i.csv' % (i,) for i in range(0, 360, 45)
    ]

    #Load data from experiments
    kw2015_fig2_x = sp.zeros((len(csvfiles), 16))
    kw2015_fig2_y = sp.zeros((len(csvfiles), 16))
    for idx, csv in enumerate(csvfiles):
        kw2015_fig2_x[idx], kw2015_fig2_y[idx] = \
            sp.genfromtxt(csv, delimiter=',')[1:].T

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
    extra_vars['gt_x'] = kw2015_fig2_x
    extra_vars['f4_stimuli_file'] = defaults.f4_stimuli_file
    extra_vars['return_var'] = 'I'
    extra_vars['precalculated_x'] = True
    extra_vars['aux_y'] = []
    extra_vars['percent_reg_train'] = 80.

    if initialize_model:
        create_stims(extra_vars)
    stim_files = np.load(extra_vars['f4_stimuli_file'])
    extra_vars['stims_all_lms'] = stim_files['stims_all_lms']

    #Run model
    #cx.run(so_all, from_gpu=False)
    #sx_all[:] = cx.Y.get()[:, :, size//2, size//2]
    adj_gt = np.mean(kw2015_fig2_y,axis=0)
    im = stim_files['so_ind'].reshape(
        n_t_hues*n_s_hues,len(_DEFAULT_KW2015_SO_PARAMETERS['norm_channels']),size,size)
    extra_vars['aux_data'] = stim_files['so_all'].transpose(0,2,3,1) 
    extra_vars['cs_hue_diff'] = stim_files['cs_hue_diff']

    optimize_model(im,adj_gt,extra_vars,defaults)

if __name__ == '__main__':
    run()
