import numpy as np
import tensorflow as tf
from collections import namedtuple


class PaperDefaults(object):
    """ Returns a simulator for the Mely-Serre circuit that runs on CUDA. """
    def __init__(self):
        self._DEFAULT_FLOATX_NP = np.float32
        self._DEFAULT_FLOATX_TF = tf.float32
        self._DEFAULT_VERBOSE = True
        self._DEFAULT_KEEPTIME = False
        self._DEFAULT_MAXITER = 100 # safe value: 100
        self._DEFAULT_STEPSIZE = 3.0 # safe value: 1.0
        self._PARAMETER_SET_VERSION = 'paper'
        self._DATADIR = '/home/drew/Documents/dmely_hmax/models/ucircuits/contextual/data' 
        self._WORKDIR = '/home/drew/Documents/dmely_hmax/models/ucircuits/contextual/working'
        self._FIGURES = '/home/drew/Documents/contextual_model/figures'
        if self._PARAMETER_SET_VERSION == 'paper':
            self._SRF = 1
            self._SSN = 9
            self._SSF = 29
            self._K_P = 1.00
            self._K_T = 1.00
        elif self.PARAMETER_SET_VERSION == 'v2':
            self._SRF = 3
            self._SSN = 9
            self._SSF = 21
            self._K_P = 1.00
            self._K_T = 1.50
        else:
            raise ValueError('Invalid value for _PARAMETER_SET_VERSION')


        self.table_name = 'F_MANUSCRIPT_FINAL'  # 'F_MANUSCRIPT_FINAL'  # 'final_full_hpcombos'
        #Figure specific items
        self.f4_stimuli_file = '/home/drew/Documents/tf_experiments/experiments/contextual_circuit/ops/special_figure_data/f4.npz' 
        self.f7_stimuli_file = '/home/drew/Documents/tf_experiments/experiments/contextual_circuit/ops/special_figure_data/f7.npy'

        ###
        self._DEFAULT_PARAMETERS = {
            'tau':          6.00,        # X: time constant
            'sigma':        0.50,        # X: saturation/decay constant
            'eta':          6.00,        # Y: time constant
            'epsilon':      0.50,        # Y: saturation/decay constant
            'xi':           4.50,        # L -> X baseline afferent strength  -- lesion non-linear inhibiyion with a 0 here
            'zeta':         0.00,        # X -> Y supplementary afferent excitation
            'gamma':        1.00 * self._K_P, # P strength [tuned summation]
            'delta':        1.00,        # Q strength [untuned summation]
            'alpha':        1.00,        # U strength (linear)
            'mu':           1.00,        # U strength (constant)
            'beta':         3.00 * self._K_T, # T strength (linear)
            'nu':           0.30 * self._K_T, # T strength (constant)
            'srf':          self._SRF,        # extent of cRF (i.e., minimal response field)
            'ssn':          self._SSN,        # extent of near surround
            'ssf':          self._SSF,        # extent of far surround
            'omega':        0.15,        # spread of weights for supp. aff. exc.
            'continuous':   True,        # feature space is continuously parametrized?
            'strides': [1, 1, 1, 1]
        }

        self.hp_optim_type = 'random_exp'  #'none' 'uniform' 'random' random_exp and 'random_linear'
        # self.lesions = ['None']  #['mely']
        # self.lesions = ['tuning']  # ,'Q','U','P','T']  #['mely']
        # self.lesions = ['None', 'Q', 'U']  # , 'omega']
        self.lesions = ['alpha_beta']  # , 'omega']
        # self.lesions = ['None', 'P', 'T', 'mu_nu', 'alpha_beta', 'crf_ecrf_combo', 'final_ecrf_overlap']  # , 'omega']
        if 'omega' in self.lesions:
            self.optimize_omega = True
            self.tunable_params = ['alpha','beta','mu','nu','gamma','delta','omega']
        else:
            self.optimize_omega = False
            self.tunable_params = ['alpha','beta','mu','nu','gamma','delta']  # ,'omega']
        self.gaussian_spatial = False  # Use gaussian connectivity in eCRFs (random sigma)
        self.gaussian_channel = False  # Use gaussian connectivity in eCRFs (random sigma)
        self.overlap_CRF_eCRF = False
        self.overlap_eCRFs = False
        self.tune_max_scale = np.repeat(100, len(self.tunable_params)) #not used with random_log
        self.tune_step = np.repeat(0.01, len(self.tunable_params)) #not used with random_log
        self.iterations = 1000
        self.db_schema = 'ops/db_schema.txt'
        self.db_problem_columns = ['f3a','f3b','f4','f5','f7','bw','tbtcso','tbp'] #['f6']#
        self.remove_figures = None  # leave to None if all
        # self.remove_figures = ['f3a','f3b','f4','f5','f7'] #['f6']#
        # self.remove_figures = ['f7','bw','tbp','tbtcso','f3a'] #['f6']#
        # self.remove_figures = ['f3b', 'f4', 'bw', 'tbp']   # for grant
        self.gpu_processes = [0, 2, 3] #run daemons on these gpus
        self.chart_style = 'hinton' # or sebs

        #### for pachaya's experiments
        self.pachaya = None # set to None if not pachaya

        ####
        self.CircuitParameters = namedtuple('CircuitParameters',
            self._DEFAULT_PARAMETERS.keys(), verbose=False, rename=False)
