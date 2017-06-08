import sys
import numpy as np
import scipy as sp
from scipy import stats
import tensorflow as tf
from copy import deepcopy
sys.path.append('../')
from parameter_defaults import PaperDefaults
from tf_helper_functions import *
from collections import namedtuple
from hmax.tools.utils import pb

defaults = PaperDefaults().__dict__
module = sys.modules[__name__] #add to the global namespace
for name, value in defaults.iteritems():
    setattr(module, name, value)
CircuitParameters = namedtuple('CircuitParameters',
    defaults['_DEFAULT_PARAMETERS'].keys(), verbose=False, rename=False)
_DEFAULT_PARAMETERS_TEMPLATE = deepcopy(defaults['_DEFAULT_PARAMETERS'])

#------------------------------------------------------------#
def _sgw(k, s):
    """ Shifted histogram of Gaussian weights, centered appropriately """
    x = sp.linspace(0.0, 1.0, k)
    if s == sp.inf:
        w = sp.ones((k,)) / float(k)
    else:
        w = stats.norm.pdf(x, loc=x[k//2], scale=s)
    return sp.roll(w / w.sum(), shift=int(sp.ceil(k/2.0)))

def _sdw(k, s):
    """ Shifted histogram of discontinuous weights, centered appropriately """
    g1 = _sgw(k=k, s=s).max()
    g2 = (1.0 - g1) / (k - 1)
    return sp.array([g1] + [g2] * (k- 1))

#------------------------------------------------------------#
class ContextualCircuit(object):
    """ Returns a simulator for the Mely-Serre circuit that runs on CUDA. """

    def __init__(self, input_shape=None, maxiter=None, 
        stepsize=None, parameters=None, keepvars=None):
        """ Initialize contexts if needed """

        self.floatXtf = _DEFAULT_FLOATX_TF
        self.floatXnp = _DEFAULT_FLOATX_NP
        self.maxiter = maxiter
        self.stepsize = stepsize
        self.keepvars = keepvars
        self.input_shape = input_shape

        try:
            for pkey, pval in parameters.iteritems():
                _DEFAULT_PARAMETERS_TEMPLATE[pkey] = pval
        except AttributeError:
            pass
        finally:
            self.parameters = CircuitParameters(
                **_DEFAULT_PARAMETERS_TEMPLATE)

        if self.maxiter is None:
            self.maxiter = _DEFAULT_MAXITER
        if self.stepsize is None:
            self.stepsize = _DEFAULT_STEPSIZE

        if self.input_shape is not None:
            self._sanity_check()

        # if input shape is known, initialize now
        if self.input_shape is not None:
            self._prepare_tensors()

    #------------------------------------------------------------#
    def _sanity_check(self):
        """ Make sure the input makes sense """
        try:
            n, h, w, k = self.input_shape
        except ValueError:
            raise ValueError('Input array must be 4-tensor')
        srf = self.parameters.srf
        ssn = self.parameters.ssn
        ssf = self.parameters.ssf

        assert ssf < h
        assert ssf < w
        assert srf < ssn < ssf
        assert self.maxiter > 0
        assert self.stepsize > 0

    #------------------------------------------------------------#
    def _prepare_tensors(self):
        """ Allocate buffer space on the GPU, etc. """
        
        n, h, w, k = self.input_shape
        SRF = self.parameters.srf
        SSN = self.parameters.ssn
        SSF = self.parameters.ssf
        OMEGA = self.parameters.omega
        ISCONTINUOUS = self.parameters.continuous

        if self.keepvars:
            self.T_t = tf.get_variable(name='T_t',dtype=self.floatXtf,shape=None)
            self.U_t = tf.get_variable(name='U_t',dtype=self.floatXtf,shape=None)
            self.P_t = tf.get_variable(name='P_t',dtype=self.floatXtf,shape=None)
            self.Q_t = tf.get_variable(name='Q_t',dtype=self.floatXtf,shape=None)

        # broadly-tuned summation
        #########################
        if self.parameters.omega:
            weights = _sgw(k=k, s=OMEGA) \
                if ISCONTINUOUS else _sdw(k=k, s=OMEGA)
            q_array = sp.array([sp.roll(weights,
                shift=shift) for shift in range(k)])
            q_array.shape = (1, 1, k, k)
            self._gpu_q = tf.get_variable(name='gpu_q',dtype=self.floatXtf,initializer=q_array.astype(self.floatXnp))

        # untuned suppression: reduction across feature axis
        ####################################################
        u_init = 1.0/k * sp.ones((1, 1, k, 1))
        self._gpu_u = tf.get_variable(name='u',dtype=self.floatXtf,initializer=u_init.astype(self.floatXnp))

        # tuned summation: pooling in h, w dimensions
        #############################################
        SSN_ = 2 * ifloor(SSN/2.0) + 1
        p_array = sp.zeros((k, k, SSN_, SSN_))

        # Uniform weights
        #----------------
        for pdx in range(k):
            p_array[pdx, pdx, :SSN, :SSN] = 1.0
        p_array[:, :, # exclude classical receptive field!
            SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0),
            SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0)] = 0.0
        p_array /= SSN**2 - SRF**2 # normalize to get true average
        p_array = p_array.transpose(2,3,0,1)

        # Gaussian weights
        #-----------------
        self._gpu_p = tf.get_variable(name='p_array',dtype=self.floatXtf,initializer=p_array.astype(self.floatXnp))

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        SSF_ = 2 * ifloor(SSF/2.0) + 1
        t_array = sp.zeros((k, k, SSF_, SSF_))

        # Uniform weights
        #----------------
        for tdx in range(k):
            t_array[tdx, tdx, :SSF, :SSF] = 1.0
        t_array[:, :, # exclude near surround!
            SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0),
            SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0)] = 0.0
        t_array /= SSF**2 - SSN**2 # normalize to get true average
        t_array = t_array.transpose(2,3,0,1)

        # Tf dimension reordering
        # Gaussian weights
        self._gpu_t = tf.get_variable(name='t_array',dtype=self.floatXtf,initializer=t_array.astype(self.floatXnp))

    #------------------------------------------------------------#
    def body(self,i0,O,I):
        U = tf.nn.conv2d(O,self._gpu_u,self.parameters.strides,padding='SAME')
        T = tf.nn.conv2d(O,self._gpu_t,self.parameters.strides,padding='SAME')
        P = tf.nn.conv2d(I,self._gpu_p,self.parameters.strides,padding='SAME')
        Q = tf.nn.conv2d(I,self._gpu_q,self.parameters.strides,padding='SAME')

        I_summand = tf.nn.relu((self.parameters.xi * self.X)
            - ((self.parameters.alpha * I + self.parameters.mu) * U)
            - ((self.parameters.beta * I + self.parameters.nu) * T))

        I = self.tf_eps_eta * I + self.tf_eta * I_summand

        O_summand = tf.nn.relu(self.parameters.zeta * I
            + self.parameters.gamma * P
            + self.parameters.delta * Q)
        O = self.tf_sig_tau * O + self.tf_tau * O_summand
        i0 += 1
        return i0,O,I

    def condition(self,i0,O,I):
        return i0 < self.maxiter

    def run(self, in_array):
        """ Do numerical integration"""
        assert in_array.ndim == 4
        if self.input_shape is None:
            self.input_shape = in_array.shape
            self._sanity_check()
            self._prepare_tensors()

        ETA = self.parameters.eta
        TAU = self.parameters.tau
        EPSILON = self.parameters.epsilon
        SIGMA = self.parameters.sigma
        self.tf_eta = tf.get_variable(name='h/eta',dtype=self.floatXtf,
            initializer=np.array(self.stepsize * 1.0/ETA).astype(self.floatXnp))
        self.tf_eps_eta = tf.get_variable(name='eps_h/eta',dtype=self.floatXtf,
            initializer=np.array(1.0 - EPSILON**2 * self.stepsize * 1.0/ETA).astype(self.floatXnp))
        self.tf_tau = tf.get_variable(name='h/tau',dtype=self.floatXtf,
            initializer=np.array(self.stepsize * 1.0/TAU).astype(self.floatXnp))
        self.tf_sig_tau = tf.get_variable(name='sig_h/tau',dtype=self.floatXtf,
            initializer=np.array(1.0 - SIGMA**2 * self.stepsize * 1.0/TAU).astype(self.floatXnp))
        
        # load copies of input into GPU
        self.X = tf.placeholder(name='input',dtype=self.floatXtf,shape=self.input_shape)

        #Using run_reference implementation
        O = tf.identity(self.X)
        I = tf.identity(self.X)
        i0 = tf.constant(0)
        elems = [i0,O,I]
        returned = tf.while_loop(self.condition, self.body, loop_vars=elems, 
            back_prop=False,swap_memory=False)#,parallel_iterations=parallel_iterations

        #Prepare output
        self.out_O = returned[1]
        self.out_I = returned[2]