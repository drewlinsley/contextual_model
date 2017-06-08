from __future__ import absolute_import
import sys
import os
import numpy as np
import scipy as sp
sys.path.append('/home/drew/Documents/')
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from dmely_hmax.models.ucircuits.contextual import stimuli as stim
from dmely_hmax.tools.utils import iround
from ops.parameter_defaults import PaperDefaults
from ops.single_hp_optim import optimize_model


def loop_operation(x, op, axis=0):
    for idx, i in enumerate(x):
        if idx == 0:
            out = loop_channel(np.squeeze(i), op)[None, :, :, :]
        else:
            out = np.append(
                out, loop_channel(np.squeeze(i), op)[None, :, :, :], axis=axis)
    return out


def loop_channel(x, op, axis=-1):
    for idx in range(x.shape[axis]):
        if idx == 0:
            out = op(x[:, :, idx])[:, :, None]
        else:
            out = np.append(out, op(x[:, :, idx])[:, :, None], axis=axis)
    return out


def normalize(x):
    return loop_operation(x, op=lambda z: (z - np.mean(z)) / np.std(z))


def resize(x, target_size):
    return loop_operation(x, op=lambda z: sp.misc.imresize(z, target_size))


def run(create_stim=True):
    defaults = PaperDefaults()

    # David's globals
    _DEFAULT_TILTEFFECT_DEGPERPIX = .25  # <OToole77>
    _DEFAULT_TILTEFFECT_SIZE = 51  # 101
    _DEFAULT_TILTEFFECT_CSIZE = iround(2. / _DEFAULT_TILTEFFECT_DEGPERPIX)
    _DEFAULT_TILTEFFECT_CVAL = .5
    _DEFAULT_TILTEFFECT_SCALES = {'ow77': 0.40, 'ms79': 0.60}  # 0.45
    _DEFAULT_TILTEFFECT_NPOINTS = 25  # 100
    _DEFAULT_TILTEFFECT_DECODER_TYPE = 'circular_vote'

    # experiment parameters
    cpt = (_DEFAULT_TILTEFFECT_SIZE//2, _DEFAULT_TILTEFFECT_SIZE//2)
    spt = (
        _DEFAULT_TILTEFFECT_SIZE//2,
        _DEFAULT_TILTEFFECT_SIZE//2 + _DEFAULT_TILTEFFECT_CSIZE)

    # simulate populations
    fl = 'conv2_2'
    if create_stim:
        sys.path.append('../../')
        from MIRC_tests import features_vgg16
        im, im_names = features_vgg16.baseline_vgg16(
            images='/home/drew/Desktop/nrsa_png', num_images=25,
            feature_layer='content_vgg/' + fl + '/Relu:0', im_ext='.png', batch_size=1)
        target_size = [400, 400]# [_DEFAULT_TILTEFFECT_SIZE, _DEFAULT_TILTEFFECT_SIZE]
        im = normalize(resize(im, target_size=target_size).astype(float))
        np.save('raw_' + fl, im)

    extra_vars = {}
    extra_vars['scale'] = _DEFAULT_TILTEFFECT_SCALES['ow77']
    extra_vars['decoder'] = _DEFAULT_TILTEFFECT_DECODER_TYPE
    extra_vars['npoints'] = _DEFAULT_TILTEFFECT_NPOINTS
    extra_vars['npoints'] = _DEFAULT_TILTEFFECT_NPOINTS
    extra_vars['cval'] = _DEFAULT_TILTEFFECT_CVAL
    extra_vars['cpt'] = cpt
    extra_vars['spt'] = spt
    extra_vars['kind'] = 'circular'
    extra_vars['figure_name'] = 'cnn_features'
    extra_vars['return_var'] = 'O'
    extra_vars['save_file'] = 'proc_' + fl
    optimize_model(im, None, extra_vars, defaults)


if __name__ == '__main__':
    run()
