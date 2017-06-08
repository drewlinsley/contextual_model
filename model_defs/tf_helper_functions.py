import numpy as np
import tensorflow as tf

def _sgw(k, s):
    """ Shifted histogram of Gaussian weights, centered appropriately """
    x = sp.linspace(0.0, 1.0, k)
    w = stats.norm.pdf(x, loc=x[k//2], scale=s)
    return tf_roll(w / w.sum(), shift=int(sp.ceil(k/2.0)))

def _tsgw(k, s):
    x = tf.linspace(0.0, 1.0, k)
    w = tf.contrib.distributions.Normal(x, mu=x[k//2], sigma=s)
    return 

def tf_roll(a,shift,axis=None):
    if axis is None:
        n = a.get_shape()[0]
        reshape = True
    else:
        try:
            n = a.get_shape()[axis]
        except IndexError:
            raise ValueError('axis must be >= 0 and < %d' % len(a.get_shape()))
        reshape = False
    if n == 0:
        return a
    shift %= n
    indexes = tf.concatenate((tf.range(n - shift, n), tf.range(n - shift)))
    res = tf.gather_nd(indexes, axis)
    if reshape:
        res = res.reshape(a.shape)
    return res

def _sdw(k, s):
    """ Shifted histogram of discontinuous weights, centered appropriately """
    g1 = _sgw(k=k, s=s).max()
    g2 = (1.0 - g1) / (k - 1)
    return sp.array([g1] + [g2] * (k- 1))

#------------------------------------------------------------------------------#
def ifloor(x):
    return np.floor(x).astype(int)

#------------------------------------------------------------------------------#
def iceil(x):
    return np.ceil(x).astype(int)

#------------------------------------------------------------------------------#
def iround(x):
    return np.around(x).astype(int)
