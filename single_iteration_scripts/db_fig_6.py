from __future__ import absolute_import
import sys,os
sys.path.append('/home/drew/Documents/')
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import scipy as sp
from scipy import signal
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_f6_hp_optim import optimize_model

def run():
    defaults = PaperDefaults()

    #David's globals
    _DEFAULT_MURAKAMISHIMOJO96_DPP = 1.0/10.    # _PARAMETER_SET_VERSION == 'paper'
    _DEFAULT_MURAKAMISHIMOJO96_DPP = .125       # _PARAMETER_SET_VERSION == 'v2'

    _DEFAULT_MURAKAMISHIMOJO96_SIZE = 201 # 51 <works>
    _DEFAULT_MURAKAMISHIMOJO96_NTRIALS = 25 # 20 <works>
    _DEFAULT_MURAKAMISHIMOJO96_WS = sp.unique(sp.around(
        1.0 / _DEFAULT_MURAKAMISHIMOJO96_DPP * \
        sp.array([0.5, 0.75, 1., 1.5, 2., 3., 4., 5., 8.]))) # <paper> <FULL3_LARGE>
    _DEFAULT_MURAKAMISHIMOJO96_RDD = 0.50 # <works> | sp.pi**2/72 # <paper>
    _DEFAULT_MURAKAMISHIMOJO96_NUNITS = 32
    _DEFAULT_MURAKAMISHIMOJO96_NDIRS = 10
    _DEFAULT_MURAKAMISHIMOJO96_NCOH = 21
    _DEFAULT_MURAKAMISHIMOJO96_NCOH4FIT = 101
    _DEFAULT_MURAKAMISHIMOJO96_SCALE = 1.0
    _DEFAULT_MURAKAMISHIMOJO96_VALUE_UP = .80
    _DEFAULT_MURAKAMISHIMOJO96_VALUE_DOWN = .30
    _DEFAULT_MURAKAMISHIMOJO96_CSVS = {
        '%s' % (subj,): [os.path.join(defaults._DATADIR, 'MS96_%s_%g.csv') % (subj, deg) \
    for deg in [0, 2, 3, 4.5, 6, 9]] for subj in ['IM', 'SM']}

    # experiment parameters
    w_paper, PSE_paper = [], [],
    w_paper_subs, PSE_paper_subs = [], []
    for idx, subject in enumerate(_DEFAULT_MURAKAMISHIMOJO96_CSVS.values()):
        for jdx, csvfile in enumerate(subject):
            w_, pse_ = sp.genfromtxt(csvfile, delimiter=',').T
            w_paper += w_.tolist()
            PSE_paper += pse_.tolist()
            w_paper_subs.append(w_)
            PSE_paper_subs.append(pse_)
    gt_data = np.zeros((len(PSE_paper_subs),len(PSE_paper_subs[0])));
    max_size = np.max(np.asarray([len(x) for x in PSE_paper_subs]))

    #Assign to 9 bins np.concatenate(w_paper_subs)
    #Take the average within each bin as the GT

    for idx,g in enumerate(PSE_paper_subs):
        if len(g) < max_size:
            g = signal.resample(g,max_size)
        gt_data[idx,:] = g
    gt = np.mean(gt_data,axis=0)
    #
    extra_vars = {}
    extra_vars['ws'] = _DEFAULT_MURAKAMISHIMOJO96_WS 
    extra_vars['rdd'] = _DEFAULT_MURAKAMISHIMOJO96_RDD
    extra_vars['size'] = _DEFAULT_MURAKAMISHIMOJO96_SIZE
    extra_vars['ntrials'] = _DEFAULT_MURAKAMISHIMOJO96_NTRIALS
    extra_vars['nunits'] = _DEFAULT_MURAKAMISHIMOJO96_NUNITS
    extra_vars['ncoh'] = _DEFAULT_MURAKAMISHIMOJO96_NCOH
    extra_vars['scale'] = _DEFAULT_MURAKAMISHIMOJO96_SCALE
    extra_vars['ndirs'] = _DEFAULT_MURAKAMISHIMOJO96_NDIRS
    extra_vars['value_up'] = _DEFAULT_MURAKAMISHIMOJO96_VALUE_UP
    extra_vars['value_down'] = _DEFAULT_MURAKAMISHIMOJO96_VALUE_DOWN
    extra_vars['ncoh4fit'] = _DEFAULT_MURAKAMISHIMOJO96_NCOH4FIT
    extra_vars['kind'] = 'circular'
    extra_vars['figure_name'] = 'f6'
    extra_vars['return_var'] = 'O'
    extra_vars['hp_file'] = os.path.join(defaults._FIGURES, 'best_hps.npz')

    optimize_model(gt,extra_vars,defaults)

if __name__ == '__main__':
    run()
