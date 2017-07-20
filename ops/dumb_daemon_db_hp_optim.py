import sys
import numpy as np
import tensorflow as tf
import model_utils
from copy import deepcopy
from timeit import default_timer as timer
from model_defs.model_cpu_port_scan_optim import ContextualCircuit, _sgw, _sdw
import scipy as sp
from ops.db_utils import update_data, get_lesion_rows_from_db, count_sets
from model_utils import iceil
from ops.parameter_defaults import PaperDefaults
import ops.fig_4_utils as f4

defaults = PaperDefaults().__dict__

def adjust_parameters(defaults,hps):
    hps_keys = hps.keys()
    for k in defaults._DEFAULT_PARAMETERS.iteritems():
        if k in hps_keys:
            defaults._DEFAULT_PARAMETERS[k] = hps[k]
    return defaults
    
def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, bar_length = 100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = ' ' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def compute_shifts(x, sess, ctx, extra_vars, default_parameters):
    start = timer()
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))  # depreciated

    # tf.group(tf.global_variables_initializer()) 
    feed_dict = {ctx.X:x,
    # ctx.xi:default_parameters._DEFAULT_PARAMETERS['xi'].reshape(-1,1),
    ctx.alpha:default_parameters._DEFAULT_PARAMETERS['alpha'].reshape(-1,1),
    ctx.beta:default_parameters._DEFAULT_PARAMETERS['beta'].reshape(-1,1),
    ctx.mu:default_parameters._DEFAULT_PARAMETERS['mu'].reshape(-1,1),
    ctx.nu:default_parameters._DEFAULT_PARAMETERS['nu'].reshape(-1,1),
    ctx.gamma:default_parameters._DEFAULT_PARAMETERS['gamma'].reshape(-1,1),
    ctx.delta:default_parameters._DEFAULT_PARAMETERS['delta'].reshape(-1,1)}

    if default_parameters.optimize_omega:
        _, _, _, k = x.shape
        weights = _sgw(k=k, s=default_parameters._DEFAULT_PARAMETERS['omega']) \
            if defaults['_DEFAULT_PARAMETERS']['continuous'] else _sdw(k=k, s=default_parameters._DEFAULT_PARAMETERS['omega'])
        q_array = sp.array(
            [sp.roll(weights, shift=shift) for shift in range(k)])
        q_array.shape = (1, 1, k, k)
        feed_dict[ctx._gpu_q] = q_array
    if extra_vars['return_var'] == 'I':
        y = sess.run(ctx.out_I,feed_dict=feed_dict)
    elif extra_vars['return_var'] == 'O':
        y = sess.run(ctx.out_O,feed_dict=feed_dict)
    end = timer()
    run_time = end - start    
    return y,run_time

def prepare_hps(parameters,hp_set):
    new_parameters = deepcopy(parameters)
    for key in parameters.tunable_params:
        if hp_set is None:
            new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(new_parameters._DEFAULT_PARAMETERS[key])
        else:
            new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(hp_set[key])
    return new_parameters

def get_fits(y,gt,extra_vars):
    if extra_vars['figure_name'] == 'tbp':
        tcs = np.zeros((2,len(extra_vars['vals'])))
        for i in range(len(extra_vars['vals'])):
            tcs[0,i] = np.corrcoef(y[0, i][::iceil(extra_vars['npoints']/float(len(extra_vars['vals'])))],gt[0][i,1:])[0,1]
            tcs[1,i] = np.corrcoef(y[1, i][::iceil(extra_vars['npoints']/float(len(extra_vars['vals'])))],gt[1][i,1:])[0,1]
        it_score = np.mean(tcs)# ** 2)
    elif extra_vars['figure_name'] == 'tbtcso':
        tcs = np.zeros((y.shape[0]))
        for i in range(y.shape[0]):
            tcs[i] = np.corrcoef(y[i,:],gt[0][i,:])[0,1]
        it_score = np.mean(tcs)# ** 2)
    elif extra_vars['figure_name'] == 'bw':
        tcs = np.zeros((y.shape[0],y.shape[1]))
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                tcs[r,c] = np.corrcoef(np.squeeze(y[r,c,:]),np.squeeze(gt[1][r,c,:]))[0,1] ** 2
        it_score = np.nanmean(tcs)
    elif extra_vars['figure_name'] == 'tbtcso':
        rs = np.zeros((gt.shape[0]))
        for i,r in enumerate(gt):
            rs[i] = np.corrcoef(y,r)
        it_score = np.mean(rs)
    else:
        it_score = np.corrcoef(y,gt)[0,1]# ** 2
    return it_score


def sampler(x):
        return abs(np.random.uniform(low=x - 1, high=x + 1) + x) ** np.random.uniform(low=-2.,high=2.)  # previously did [0, 2]


def optimize_model(im,gt,extra_vars,parameters):
 
    #Do outer loop of lesions -- embed in variable scopes
    for lesion in parameters.lesions:
        outer_parameters = deepcopy(parameters)
        outer_parameters.lesions = lesion
        #Optional stuff and data prep
        if extra_vars['figure_name'] == 'f4': #precalculated
            stim_files = f4.create_stims(extra_vars)
            extra_vars['stims_all_lms'] = stim_files['stims_all_lms']
            im = stim_files['so_ind'].reshape(
                extra_vars['n_t_hues']*extra_vars['n_s_hues'],len(extra_vars['_DEFAULT_KW2015_SO_PARAMETERS']['norm_channels']),extra_vars['size'],extra_vars['size'])
            extra_vars['aux_data'] = stim_files['so_all'].transpose(0,2,3,1)
            extra_vars['cs_hue_diff'] = stim_files['cs_hue_diff']
            x = im.transpose(0,2,3,1) #need to copy or what?
            with tf.device('/gpu:0'):
                with tf.variable_scope('aux_' + lesion):
                    aux_ctx = ContextualCircuit(lesions=lesion)
                    aux_ctx.run(extra_vars['aux_data']) #builds tf graph with shape of x
        elif extra_vars['figure_name'] == 'tbp' or extra_vars['figure_name'] == 'tbtcso' or extra_vars['figure_name'] == 'bw' or extra_vars['figure_name'] == 'cross_orientation_suppression':
            x = im.transpose(0,2,3,1)
        else:
            extra_vars['scale'] = sampler(extra_vars['scale'])
            x = model_utils.get_population2(im,
                npoints=extra_vars['npoints'], kind=extra_vars['kind'],
                 scale=extra_vars['scale']).astype(outer_parameters._DEFAULT_FLOATX_NP).transpose(0,2,3,1) #transpose to bhwc

        # Build main graph
        with tf.device('/gpu:0'):
            with tf.variable_scope('main_' + lesion):
                ctx = ContextualCircuit(lesions=outer_parameters.lesions)
                ctx.run(x)  # builds tf graph with shape of x

        # Do hp optimization
        num_sets = count_sets(lesion, extra_vars['figure_name'])[0]['count']
        if num_sets > 0:
            idx = 0  # keep a count
            # Initialize the session
            with tf.Session(
                    config=tf.ConfigProto(
                        allow_soft_placement=True)) as sess:
                while 1:  # while we have hp to run on this lesion
                    hp_set = get_lesion_rows_from_db(
                        lesion, extra_vars['figure_name'], get_one=True)
                    if hp_set is None and parameters.pachaya is not True:
                        break
                    else:
                        if parameters.pachaya:
                            random_parameters = prepare_hps(
                                outer_parameters, None)
                        else:
                            random_parameters = prepare_hps(
                                outer_parameters, hp_set)

                        # Special case for fig 4
                        if extra_vars['figure_name'] == 'f4':
                            extra_vars['aux_y'], _ = compute_shifts(
                                x=extra_vars['aux_data'], sess=sess,
                                ctx=aux_ctx, extra_vars=extra_vars,
                                default_parameters=random_parameters)
                        # Apply the model to the data
                        oy, run_time = compute_shifts(
                            x=x, sess=sess, ctx=ctx,
                            extra_vars=extra_vars,
                            default_parameters=random_parameters) 

                        # Postprocess the model's outputs
                        y, aux_data = model_utils.data_postprocessing(
                            x, oy, extra_vars)  # maybe return a dict instead?

                        # Correlate simulation and ground truth
                        it_score = get_fits(y, gt, extra_vars)

                        # Add to database
                        update_data(
                            random_parameters, extra_vars['figure_name'],
                            hp_set['_id'], it_score)
                        printProgress(
                            idx, num_sets,
                            prefix=extra_vars['figure_name'] +
                            ' progress on lesion ' + lesion + ':',
                            suffix='Iteration time: ' +
                            str(np.around(run_time, 2)) +
                            '; Correlation: ' + str(np.around(it_score, 2)),
                            bar_length=30)
                    idx += 1
                    break
