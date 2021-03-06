import sys, os
import numpy as np
import scipy as sp
sys.path.append('/home/drew/Documents/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import tensorflow as tf
import model_utils
import model_cutils as cutils
from tqdm import tqdm
from copy import deepcopy
from timeit import default_timer as timer
from model_defs.model_cpu_port_scan_optim import ContextualCircuit
from db_utils import update_data, gather_data
from skimage.color import xyz2rgb, rgb2xyz

CIECAM02_CAT = sp.array([[ 0.7328,  0.4296, -0.1624],
                         [-0.7036,  1.6975,  0.0061],
                         [ 0.003 ,  0.0136,  0.9834]])

def adjust_parameters(defaults,hps):
    if hps is not None:
        hps_keys = hps.keys()
        for k in defaults._DEFAULT_PARAMETERS.iterkeys():
            if k in hps_keys:
                defaults._DEFAULT_PARAMETERS[k] = hps[k]
    defaults.lesions = hps['lesions']
    defaults.current_figure = hps['current_figure']
    defaults._id = hps['_id']
    return defaults
    
def _map_responses_on_cycles(size, csize, ssize, cps,
    cval, nval, fval, emptyval=None):
    """ Generate stimuli closer to Shevell & Monnier """

    assert len(cval.ravel()) == len(nval.ravel()) == len(fval.ravel())

    nc = len(cval.ravel())
    im = sp.zeros((nc, size, size))
    jm = _get_cyclical_stimuli_linear(size, csize, ssize, cps)

    im[:, jm ==  0] = cval.ravel().reshape((nc, 1))
    im[:, jm ==  1] = nval.ravel().reshape((nc, 1))
    im[:, jm == -1] = fval.ravel().reshape((nc, 1))
    if emptyval is not None:
        im[:, sp.isnan(jm)] = emptyval.ravel((nc, 1))
    else:
        im[:, sp.isnan(jm)] = 0.0

    return im

def _get_cyclical_stimuli_linear(sz, csz, ssz, cps):
    """"""
    assert sz % 2 == 1
    assert cps > 0
    assert cps <= sz//2
    assert (ssz is None) or (ssz % 2 == 1)
    gx, _ = sp.mgrid[-sz//2+1:sz//2+1, -sz//2+1:sz//2+1]
    gr = sp.absolute(gx)
    bounds = sp.linspace(csz//2, sz//2, cps+1)
    for idx in range(cps):
        gr[(bounds[idx] < gr) * (gr <= bounds[idx+1])] = \
            (-1 if (idx % 2) else -2)
    gr[gr == -2] = 1
    gr[sz//2-csz//2:sz//2+csz//2+1, :] = 0
    if ssz is not None:
        gr = gr.astype(float)
        gr[sp.absolute(gx) > ssz//2] = sp.nan
    return gr


def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, bar_length = 100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = ' ' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def compute_shifts(x, ctx, extra_vars, default_parameters):
    start = timer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
        start = timer()
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        feed_dict = {ctx.X:x,
        #ctx.xi:default_parameters._DEFAULT_PARAMETERS['xi'].reshape(-1,1),
        ctx.alpha:np.reshape(default_parameters._DEFAULT_PARAMETERS['alpha'],(-1,1)),
        ctx.beta:np.reshape(default_parameters._DEFAULT_PARAMETERS['beta'],(-1,1)),
        ctx.mu:np.reshape(default_parameters._DEFAULT_PARAMETERS['mu'],(-1,1)),
        ctx.nu:np.reshape(default_parameters._DEFAULT_PARAMETERS['nu'],(-1,1)),
        ctx.gamma:np.reshape(default_parameters._DEFAULT_PARAMETERS['gamma'],(-1,1)),
        ctx.delta:np.reshape(default_parameters._DEFAULT_PARAMETERS['delta'],(-1,1))}
        if extra_vars['return_var'] == 'I':
            y = sess.run(ctx.out_I,feed_dict=feed_dict)
        elif extra_vars['return_var'] == 'O':
            y = sess.run(ctx.out_O,feed_dict=feed_dict)
        end = timer()
        run_time = end - start    
    return y,run_time

def prepare_hps(parameters,hp_sets,idx):
    new_parameters = deepcopy(parameters)
    for key in parameters.tunable_params:
        new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(hp_sets[key][idx])
    return new_parameters

def XYZ2lms(XYZimage):
    """"""
    return sp.tensordot(XYZimage, CIECAM02_CAT, axes=(-1, 1))

def rgb2lms(rgbimage):
    """"""
    return XYZ2lms(rgb2xyz(rgbimage))

def make_stimulus(col,extra_vars):
    so = {}

    sz_ = cutils._DEFAULT_SO_RFSIZE * 3
    for cname, rgb in extra_vars['_DEFAULT_SM2003_COLORS_RGB'].iteritems():
        im_rgb = sp.array([sp.zeros((sz_, sz_)) + rgb[k] \
            for k in range(3)]).swapaxes(0, -1)
        im_lms = rgb2lms(im_rgb)
        so[cname] = cutils.GET_SO_STANDARD(im_lms)[:, sz_//2, sz_//2]

    # make population responses (faster than convolution with large inputs)
    #######################################################################
    nc = len(cutils._DEFAULT_SO_CHANNELS)
    x_cyclical = sp.zeros((2*extra_vars['n_cps'], nc, extra_vars['size'], extra_vars['size']))

    for idx, cps in enumerate(extra_vars['cpss']):
        x_cyclical[idx] = _map_responses_on_cycles(
            extra_vars['size'], extra_vars['csize'], extra_vars['ssize'], cps,
            so[col], so['purple'], so['lime'])
        x_cyclical[idx + extra_vars['n_cps']] = _map_responses_on_cycles(
            extra_vars['size'], extra_vars['csize'], extra_vars['ssize'], cps,
            so[col], so['lime'], so['purple'])
    return x_cyclical.transpose(0,2,3,1)


def create_stimuli(gt,extra_vars,parameters):
    all_x = []
    for idx, col in enumerate(extra_vars['test_colors']):
        all_x.append(make_stimulus(col,extra_vars))
    np.save(parameters.f7_stimuli_file,all_x)

def optimize_model(gt,extra_vars,parameters):
    print('Running hyperparameter optimizations after lesioning: ' + ' '.join(parameters.lesions))
    build_graph = True
    stimuli = np.load(parameters.f7_stimuli_file)
    shift_phase = sp.zeros((extra_vars['n_cps'],))
    shift_anti = sp.zeros((extra_vars['n_cps'],))
    for idx, col in enumerate(extra_vars['test_colors']):
        x = stimuli[idx] #make_stimulus(col,extra_vars)
        extra_vars['x'] = x
        extra_vars['test_color'] = col
        if build_graph:
            with tf.device('/gpu:0'):
                with tf.variable_scope('main'):
                    ctx = ContextualCircuit(lesions=parameters.lesions, parameters=parameters._DEFAULT_PARAMETERS)
                    ctx.run(x) #builds tf graph with shape of x
            build_graph = False

        y, run_time = compute_shifts(x=x, ctx=ctx, 
             extra_vars=extra_vars, default_parameters=parameters)
        phase,anti = model_utils.data_postprocessing(x,y,extra_vars)
        shift_phase += phase / float(extra_vars['n_col'])
        shift_anti += anti / float(extra_vars['n_col'])

    phase_score = np.corrcoef(shift_phase,gt[0])[0,1] ** 2
    anti_score = np.corrcoef(shift_anti,gt[1])[0,1] ** 2
    it_score = np.mean([phase_score,anti_score])

    #Add to database
    update_data(parameters,extra_vars['figure_name'],parameters._id,it_score)
    return