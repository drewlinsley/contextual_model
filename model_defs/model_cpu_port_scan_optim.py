import sys
import numpy as np
import scipy as sp
from scipy import stats
import tensorflow as tf
from copy import deepcopy
sys.path.append('../')
from ops.parameter_defaults import PaperDefaults
from tf_helper_functions import *
from collections import namedtuple


defaults = PaperDefaults().__dict__
module = sys.modules[__name__]  # add to the global namespace
for name, value in defaults.iteritems():
    setattr(module, name, value)
CircuitParameters = namedtuple(
    'CircuitParameters',
    defaults['_DEFAULT_PARAMETERS'].keys(), verbose=False, rename=False)
_DEFAULT_PARAMETERS_TEMPLATE = deepcopy(defaults['_DEFAULT_PARAMETERS'])


def sampler(x):
        return abs(np.random.uniform(low=x - 1, high=x + 1) + x) ** np.random.uniform(low=-2.,high=2.)  # previously did [0, 2]


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


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
    return sp.array([g1] + [g2] * (k - 1))


class ContextualCircuit(object):
    """ Returns a simulator for the Mely-Serre circuit that runs on CUDA. """

    def __init__(
        self, input_shape=None, maxiter=None,
        stepsize=None, parameters=None, keepvars=None,
            lesions=['']):
        """ Initialize contexts if needed """

        self.floatXtf = _DEFAULT_FLOATX_TF
        self.floatXnp = _DEFAULT_FLOATX_NP
        self.maxiter = maxiter
        self.stepsize = stepsize
        self.keepvars = keepvars
        self.input_shape = input_shape
        self.lesions = lesions
        self.gaussian_spatial = gaussian_spatial
        self.gaussian_channel = gaussian_channel
        self.overlap_CRF_eCRF = overlap_CRF_eCRF
        self.overlap_eCRFs = overlap_eCRFs

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

        assert type(self.lesions) is str

    def _prepare_tensors(self):
        """ Allocate buffer space on the GPU, etc. """
        n, h, w, k = self.input_shape
        SRF = self.parameters.srf
        SSN = self.parameters.ssn
        SSF = self.parameters.ssf
        OMEGA = self.parameters.omega
        ISCONTINUOUS = self.parameters.continuous

        if self.keepvars:
            self.T_t = tf.get_variable(
                name='T_t',
                dtype=self.floatXtf,
                shape=None)
            self.U_t = tf.get_variable(
                name='U_t',
                dtype=self.floatXtf,
                shape=None)
            self.P_t = tf.get_variable(
                name='P_t',
                dtype=self.floatXtf,
                shape=None)
            self.Q_t = tf.get_variable(
                name='Q_t',
                dtype=self.floatXtf,
                shape=None)

        # broadly-tuned summation
        #########################
        if self.parameters.omega:
            weights = _sgw(k=k, s=OMEGA) \
                if ISCONTINUOUS else _sdw(k=k, s=OMEGA)
            q_array = sp.array(
                [
                    sp.roll(weights, shift=shift) for shift in range(k)])
            q_array.shape = (1, 1, k, k)
            if defaults['optimize_omega']:
                self._gpu_q = tf.placeholder(
                    name='gpu_q',
                    dtype=self.floatXtf,
                    shape=q_array.shape)
            else:
                self._gpu_q = tf.get_variable(
                    name='gpu_q',
                    dtype=self.floatXtf,
                    initializer=q_array.astype(self.floatXnp))

        # untuned suppression: reduction across feature axis
        ####################################################
        u_init = 1.0/k * sp.ones((1, 1, k, 1))
        self._gpu_u = tf.get_variable(
            name='u',
            dtype=self.floatXtf,
            initializer=u_init.astype(self.floatXnp))

        ###### Reviews Analysis 1: Try changing the eCRF tuning properties... 
        # Set these to gaussians instead of uniform and try a few different sigmas.
        if self.overlap_eCRFs:
            # tuned summation: pooling in h, w dimensions
            #############################################
            SSF_ = 2 * ifloor(SSF/2.0) + 1
            t_array = sp.zeros((k, k, SSF_, SSF_))
            SSN_ = 2 * ifloor(SSN/2.0) + 1
            p_array = sp.zeros((k, k, SSF_, SSF_))  # Both 29 pixel kernels

            # SSN is near SSF is far

            # Uniform weights
            for pdx in range(k):
                p_array[pdx, pdx, :SSF, :SSF] = 1.0
            p_array[
                :, :,  # exclude classical receptive field!
                SSF//2-ifloor(SRF/2.0):SSF//2+iceil(SRF/2.0),
                SSF//2-ifloor(SRF/2.0):SSF//2+iceil(SRF/2.0)] = 0.0
            p_array /= SSF**2 - SRF**2  # normalize to get true average
            p_array = p_array.transpose(2, 3, 0, 1)

            # Gaussian weights
            self._gpu_p = tf.get_variable(
                name='p_array', dtype=self.floatXtf, initializer=p_array.astype(
                    self.floatXnp))

            # tuned suppression: pooling in h, w dimensions
            ###############################################

            # Uniform weights
            for tdx in range(k):
                t_array[tdx, tdx, :SSF, :SSF] = 1.0
            t_array[
                :, :,  # exclude classical receptive field!
                SSF//2-ifloor(SRF/2.0):SSF//2+iceil(SRF/2.0),
                SSF//2-ifloor(SRF/2.0):SSF//2+iceil(SRF/2.0)] = 0.0
            t_array /= SSF**2 - SRF**2  # normalize to get true average
            t_array = t_array.transpose(2, 3, 0, 1)

            # Tf dimension reordering
            # Gaussian weights
            self._gpu_t = tf.get_variable(
                name='t_array', dtype=self.floatXtf, initializer=t_array.astype(
                    self.floatXnp))
        else:
            # tuned summation: pooling in h, w dimensions
            #############################################
            SSN_ = 2 * ifloor(SSN/2.0) + 1
            p_array = sp.zeros((k, k, SSN_, SSN_))

            # Uniform weights
            for pdx in range(k):
                p_array[pdx, pdx, :SSN, :SSN] = 1.0
            p_array[
                :, :,  # exclude classical receptive field!
                SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0),
                SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0)] = 0.0
            p_array /= SSN**2 - SRF**2  # normalize to get true average
            if self.gaussian_spatial:
                g = makeGaussian(SSN_, fwhm=sampler(SSN_))
                p_array *= g
            if self.gaussian_channel:
                for pdx in range(k):
                    p_array[:, :, :SSN, :SSN] = 1.0
                p_array[
                    :, :,  # exclude classical receptive field!
                    SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0),
                    SSN//2-ifloor(SRF/2.0):SSN//2+iceil(SRF/2.0)] = 0.0
                p_array /= SSN**2 - SRF**2  # normalize to get true average
                weights = _sgw(k=k, s=sampler(OMEGA)) \
                    if ISCONTINUOUS else _sdw(k=k, s=OMEGA)
                weight_array = sp.array(
                    [sp.roll(weights, shift=shift) for shift in range(k)])
                p_array = p_array * weight_array[:, :, None, None]

            # Tf dimension reordering
            p_array = p_array.transpose(2, 3, 0, 1)
            self._gpu_p = tf.get_variable(
                name='p_array', dtype=self.floatXtf, initializer=p_array.astype(
                    self.floatXnp))

            # tuned suppression: pooling in h, w dimensions
            ###############################################
            SSF_ = 2 * ifloor(SSF/2.0) + 1
            t_array = sp.zeros((k, k, SSF_, SSF_))

            # Uniform weights
            for tdx in range(k):
                t_array[tdx, tdx, :SSF, :SSF] = 1.0
            t_array[:, :,  # exclude near surround!
                SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0),
                SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0)] = 0.0
            t_array /= SSF**2 - SSN**2  # normalize to get true average
            if self.gaussian_spatial:
                g = makeGaussian(SSF_, fwhm=sampler(SSF_))
                t_array *= g
            if self.gaussian_channel:
                for tdx in range(k):
                    t_array[:, :, :SSF, :SSF] = 1.0
                t_array[:, :,  # exclude near surround!
                    SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0),
                    SSF//2-ifloor(SSN/2.0):SSF//2+iceil(SSN/2.0)] = 0.0
                t_array /= SSF**2 - SSN**2  # normalize to get true average
                weights = _sgw(k=k, s=sampler(OMEGA)) \
                    if ISCONTINUOUS else _sdw(k=k, s=OMEGA)
                weight_array = sp.array(
                    [sp.roll(weights, shift=shift) for shift in range(k)])
                t_array = t_array * weight_array[:, :, None, None]

            # Tf dimension reordering
            t_array = t_array.transpose(2, 3, 0, 1)
            self._gpu_t = tf.get_variable(
                name='t_array', dtype=self.floatXtf, initializer=t_array.astype(
                    self.floatXnp))

    def body(
            self, i0, O, I, alpha, beta, mu, nu, gamma, delta):  # , I_diff, O_diff):

        # Track initial I, O
        # prev_I = tf.identity(I)
        # prev_O = tf.identity(O)

        if 'U' in self.lesions:
            U = tf.constant(0.)
        else:
            U = tf.nn.conv2d(
                O, self._gpu_u, self.parameters.strides, padding='SAME')

        if 'T' in self.lesions:
            T = tf.constant(0.)
        else:
            T = tf.nn.conv2d(
                O, self._gpu_t, self.parameters.strides, padding='SAME')

        if 'P' in self.lesions:
            P = tf.constant(0.)
        else:
            P = tf.nn.conv2d(
                I, self._gpu_p, self.parameters.strides, padding='SAME')

        if 'Q' in self.lesions:
            Q = tf.constant(0.)
        else:
            Q = tf.nn.conv2d(
                I, self._gpu_q, self.parameters.strides, padding='SAME')

        I_summand = tf.nn.relu(
            (self.xi * self.X)
            - ((alpha * I + mu) * U)
            - ((beta * I + nu) * T))

        I = self.tf_eps_eta * I + self.tf_eta * I_summand

        O_summand = tf.nn.relu(
            self.parameters.zeta * I
            + gamma * P
            + delta * Q)
        O = self.tf_sig_tau * O + self.tf_tau * O_summand

        # Store deviation of I and O from previous timestep
        # I_diff = tf.concat(
        #     axis=0,
        #     values=[I_diff, tf.reshape(tf.reduce_mean(I - prev_I), [1, ])])
        # O_diff = tf.concat(
        #     axis=0,
        #     values=[O_diff, tf.reshape(tf.reduce_mean(O - prev_O), [1, ])])

        # Iterate counter
        i0 += 1
        return i0, O, I, alpha, beta, mu, nu, gamma, delta  # , I_diff, O_diff

    def body_overlap_CRF_eCRF(
            self, i0, O, I, alpha, beta, mu, nu, gamma, delta):  # , I_diff, O_diff):

        '''
        ###### Reviews Analysis 2: Lesion one of these
        Combine the CRF activities with the near eCRFs
        Q *= P
        U *= P
        excise P
        '''

        if 'U' in self.lesions:
            U = tf.constant(0.)
        else:
            U = tf.nn.conv2d(
                O, self._gpu_u, self.parameters.strides, padding='SAME')

        if 'T' in self.lesions:
            T = tf.constant(0.)
        else:
            T = tf.nn.conv2d(
                O, self._gpu_t, self.parameters.strides, padding='SAME')

        if 'P' in self.lesions:
            P = tf.constant(0.)
        else:
            P = tf.nn.conv2d(
                I, self._gpu_p, self.parameters.strides, padding='SAME')

        if 'Q' in self.lesions:
            Q = tf.constant(0.)
        else:
            Q = tf.nn.conv2d(
                I, self._gpu_q, self.parameters.strides, padding='SAME')

        I_summand = tf.nn.relu(
            (self.xi * self.X)
            - ((alpha * I + mu) * (U * P))
            - ((beta * I + nu) * T))

        I = self.tf_eps_eta * I + self.tf_eta * I_summand

        O_summand = tf.nn.relu(
            self.parameters.zeta * I
            + delta * (Q * P))
        O = self.tf_sig_tau * O + self.tf_tau * O_summand

        # Store deviation of I and O from previous timestep
        # I_diff = tf.concat(
        #     axis=0,
        #     values=[I_diff, tf.reshape(tf.reduce_mean(I - prev_I), [1, ])])
        # O_diff = tf.concat(
        #     axis=0,
        #     values=[O_diff, tf.reshape(tf.reduce_mean(O - prev_O), [1, ])])

        # Iterate counter
        i0 += 1
        return i0, O, I, alpha, beta, mu, nu, gamma, delta  # , I_diff, O_diff

    # def condition(self,i0,O,I,xi,alpha,beta,mu,nu,gamma,delta):
    def condition(
            self, i0, O, I, alpha, beta, mu, nu, gamma, delta):  # , I_diff, O_diff):
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
        self.xi = self.parameters.xi
        self.tf_eta = tf.get_variable(
            name='h/eta', dtype=self.floatXtf,
            initializer=np.array(
                self.stepsize * 1.0/ETA).astype(
                self.floatXnp))
        self.tf_eps_eta = tf.get_variable(
            name='eps_h/eta', dtype=self.floatXtf,
            initializer=np.array(
                1.0 - EPSILON**2 * self.stepsize * 1.0/ETA).astype(
                self.floatXnp))


        self.tf_tau = tf.get_variable(
            name='h/tau', dtype=self.floatXtf,
            initializer=np.array(
                self.stepsize * 1.0/TAU).astype(
                self.floatXnp))
        self.tf_sig_tau = tf.get_variable(
            name='sig_h/tau', dtype=self.floatXtf,
            initializer=np.array(
                1.0 - SIGMA**2 * self.stepsize * 1.0/TAU).astype(
                self.floatXnp))

        # load copies of input into GPU
        self.X = tf.placeholder(
            name='input', dtype=self.floatXtf, shape=self.input_shape)
        self.alpha = tf.placeholder(
            name='alpha', dtype=self.floatXtf, shape=[1, None])
        self.beta = tf.placeholder(
            name='beta', dtype=self.floatXtf, shape=[1, None])
        self.mu = tf.placeholder(
            name='mu', dtype=self.floatXtf, shape=[1, None])
        self.nu = tf.placeholder(
            name='nu', dtype=self.floatXtf, shape=[1, None])
        self.gamma = tf.placeholder(
            name='gamma', dtype=self.floatXtf, shape=[1, None])
        self.delta = tf.placeholder(
            name='delta', dtype=self.floatXtf, shape=[1, None])

        # Using run_reference implementation
        i0 = tf.constant(0)
        O = tf.identity(self.X)
        I = tf.identity(self.X)
        # I_diff = tf.zeros_like(
        #     tf.placeholder(tf.float32, shape=[1, None]),
        #     dtype=tf.float32)
        # O_diff = tf.zeros_like(
        #     tf.placeholder(tf.float32, shape=[1, None]),
        #     dtype=tf.float32)

        nl_lesions = 'ignore'  # doing this from the parameters instead
        if nl_lesions is 'l':
            print 'Ablating mu'
            self.mu = self.mu * 0
            print 'Ablating nu'
            self.nu = self.nu * 0
        elif nl_lesions is 'nl':
            print 'Ablating alpha'
            self.alpha = self.alpha * 0
            print 'Ablating beta'
            self.beta = self.beta * 0

        # While loop
        elems = [
            i0,
            O,
            I,
            self.alpha,
            self.beta,
            self.mu,
            self.nu,
            self.gamma,
            self.delta,
            # I_diff,
            # O_diff
        ]

        if self.overlap_CRF_eCRF:
            body_fun = self.body_overlap_CRF_eCRF
        else:
            body_fun = self.body

        returned = tf.while_loop(
            self.condition,
            body_fun,
            loop_vars=elems,
            back_prop=False,
            swap_memory=False)

        # Prepare output
        self.out_O = returned[1]
        self.out_I = returned[2]
        # self.I_diff = returned[-2]
        # self.O_diff = returned[-1]
