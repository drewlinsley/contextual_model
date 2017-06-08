from __future__ import absolute_import
import sys,os,joblib
sys.path.append('/home/drew/Documents/')
sys.path.append('/home/drew/Documents/tf_experiments/experiments/contextual_circuit/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy as np
import scipy as sp
from hmax.tools.utils import iround
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_f7_hp_optim import optimize_model, create_stimuli
import ops.model_cutils as cutils

def run(make_stims=False):
    defaults = PaperDefaults()

    #David's globals
    test_colors=['orange', 'turquoise']
    size=77
    csize=3
    ssize=None
    _MINPERPIX = 2.
    _DEFAULT_SM2003_CSIZE_MIN = 6.0
    _DEFAULT_SM2003_SIZE_MIN = 153.0
    _DEFAULT_SM2003_CPDS = sp.array([0., 1., 2., 3.3, 5., 10.]) # <from paper>
    _DEFAULT_SM2003_CPMS = _DEFAULT_SM2003_CPDS / 60.
    _DEFAULT_SM2003_CPSS = iround(1 + _DEFAULT_SM2003_CPMS * \
        (_DEFAULT_SM2003_SIZE_MIN - _DEFAULT_SM2003_CSIZE_MIN)/2.0) # <realistic>
    _DEFAULT_SM2003_CSIZE = iround(_DEFAULT_SM2003_CSIZE_MIN/_MINPERPIX)
    cpss=_DEFAULT_SM2003_CPSS

    #Use color parameters
    defaults._DEFAULT_PARAMETERS['continuous'] = False
    defaults._DEFAULT_PARAMETERS['srf'] = _DEFAULT_SM2003_CSIZE 

    csvfiles={
    'ObsML': {
        'PL': defaults._DATADIR + '/SM2003_Fig5_ObsML_PL.csv',
        'LP': defaults._DATADIR + '/SM2003_Fig5_ObsML_LP.csv',
    },
    'ObsPM': {
        'PL': defaults._DATADIR + '/SM2003_Fig5_ObsPM_PL.csv',
        'LP': defaults._DATADIR + '/SM2003_Fig5_ObsPM_LP.csv',
    },
    }

    _DEFAULT_SM2003_COLORS_RGB = {
        'orange':       sp.array([.99, .65, .47]),
        'purple':       sp.array([.65, .54, .77]),
            # adjusted to have Y=.30 #sp.array([.60, .49, .71]),
        'purple_alt':   sp.array([.60, .48, .70]),
        'lime':         sp.array([.56, .60, .49]),
            # adjusted to have Y=.30 #sp.array([.61, .65, .53]),
        'lime_alt':     sp.array([.51, .61, .53]),
        'turquoise':    sp.array([.44, .78, .73]),
        'neutral':      sp.array([.66, .66, .66]),
        'lilac':        sp.array([.89, .62, .80]),
        'citrus':       sp.array([.67, 1.0, .50]),
        'EEW':          sp.array([.58, .58, .58])
    }

    # O'Toole and Wenderoth (1977)
    # load digitzed data from original paper
    ########################################
    shift_phase_paper = sp.array([
        sp.genfromtxt(csvfiles['ObsML']['PL'], delimiter=',').T[1],
        sp.genfromtxt(csvfiles['ObsPM']['PL'], delimiter=',').T[1]])
    shift_anti_paper = sp.array([
        sp.genfromtxt(csvfiles['ObsML']['LP'], delimiter=',').T[1],
        sp.genfromtxt(csvfiles['ObsPM']['LP'], delimiter=',').T[1]])
    shift_phase_paper = np.mean(shift_phase_paper, axis=0) #double check this
    shift_anti_paper = np.mean(shift_anti_paper, axis=0)
    gt = np.vstack((shift_phase_paper,shift_anti_paper))

    #Also preload regs for postprocessing
    regpath = os.path.join(defaults._WORKDIR,\
        'ShevellMonnier2003.reg.pkl.std.srf%issn%issf%i' \
            % (defaults._DEFAULT_PARAMETERS['srf'], defaults._DEFAULT_PARAMETERS['ssn'], defaults._DEFAULT_PARAMETERS['ssf']))

    reg_X_SO = joblib.load(regpath)['reg_X_SO']
    reg_Y_SO = joblib.load(regpath)['reg_Y_SO']
    reg_Z_SO = joblib.load(regpath)['reg_Z_SO']
    scaler_SO = joblib.load(regpath)['scaler_SO']

    reg_X_SX = joblib.load(regpath)['reg_X_SX']
    reg_Y_SX = joblib.load(regpath)['reg_Y_SX']
    reg_Z_SX = joblib.load(regpath)['reg_Z_SX']
    scaler_SX = joblib.load(regpath)['scaler_SX']
    
    so2image = cutils.get_XYZ2RGB_predictor(reg_X_SO, reg_Y_SO, reg_Z_SO, scaler_SO)
    sx2image = cutils.get_XYZ2RGB_predictor(reg_X_SX, reg_Y_SX, reg_Z_SX, scaler_SX)

    #Add to a dictionary
    extra_vars = {}
    extra_vars['n_cps'] = len(cpss)
    extra_vars['cpss'] = cpss
    extra_vars['test_colors'] = test_colors
    extra_vars['_DEFAULT_SM2003_COLORS_RGB'] = _DEFAULT_SM2003_COLORS_RGB
    extra_vars['size'] = size
    extra_vars['csize'] = csize
    extra_vars['ssize'] = ssize
    extra_vars['kind'] = 'circular'
    extra_vars['so2image'] = so2image
    extra_vars['sx2image'] = sx2image
    extra_vars['n_col'] = len(test_colors)
    extra_vars['figure_name'] = 'f7'
    extra_vars['return_var'] = 'O'

    # measure shift for phase & antiphase inducers, for each test
    if make_stims:
        create_stimuli(gt,extra_vars,defaults)
    else:
        optimize_model(gt,extra_vars,defaults)

if __name__ == '__main__':
    run()
