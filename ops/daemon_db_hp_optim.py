import sys, os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
sys.path.append('/home/drew/Documents/')
import tensorflow as tf
import model_utils
from copy import deepcopy
from timeit import default_timer as timer
from model_defs.model_cpu_port_scan_optim import ContextualCircuit
from ops.db_utils import update_data, gather_data
from hmax.tools.utils import iceil
from ops.sampling import grab_sampler, uniform_sampling, random_sampling, random_log_sampling, pass_params

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

def get_fits(y,gt,extra_vars):
    if extra_vars['figure_name'] == 'tbp':
        tcs = np.zeros((2,len(extra_vars['vals'])))
        for i in range(len(extra_vars['vals'])):
            tcs[0,i] = np.corrcoef(y[0, i][::iceil(extra_vars['npoints']/float(len(extra_vars['vals'])))],gt[0][i,1:])[0,1]
            tcs[1,i] = np.corrcoef(y[1, i][::iceil(extra_vars['npoints']/float(len(extra_vars['vals'])))],gt[1][i,1:])[0,1]
        it_score = np.mean(tcs ** 2)
    elif extra_vars['figure_name'] == 'tbtcso':
        tcs = np.zeros((y.shape[0]))
        for i in range(y.shape[0]):
            tcs[i] = np.corrcoef(y[i,:],gt[0][i,:])[0,1]
        it_score = np.mean(tcs ** 2)
    elif extra_vars['figure_name'] == 'bw':
        tcs = np.zeros((y.shape[0],y.shape[1]))
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                tcs[r,c] = np.corrcoef(np.squeeze(y[r,c,:]),np.squeeze(gt[1][r,c,:]))[0,1] ** 2
        it_score = np.nanmean(tcs)
    else:
        it_score = np.corrcoef(y,gt)[0,1] ** 2
    return it_score

def optimize_model(im,gt,extra_vars,parameters):
    #Get data
    if extra_vars['figure_name'] == 'f4': #precalculated
        x = im.transpose(0,2,3,1) #need to copy or what?
        with tf.device('/gpu:0'):
            with tf.variable_scope('aux'):
                aux_ctx = ContextualCircuit(lesions=parameters.lesions)
                aux_ctx.run(extra_vars['aux_data']) #builds tf graph with shape of x
    elif extra_vars['figure_name'] == 'tbp' or extra_vars['figure_name'] == 'tbtcso' or extra_vars['figure_name'] == 'bw':
        x = im.transpose(0,2,3,1)
    else:
        x = model_utils.get_population2(im,
            npoints=extra_vars['npoints'], kind=extra_vars['kind'],
             scale=extra_vars['scale']).astype(parameters._DEFAULT_FLOATX_NP).transpose(0,2,3,1) #transpose to bhwc

    #Build graph
    with tf.device('/gpu:0'):
        with tf.variable_scope('main'):
            ctx = ContextualCircuit(lesions=parameters.lesions)
            ctx.run(x) #builds tf graph with shape of x

        #Special case for fig 4
        if extra_vars['figure_name'] == 'f4':
            extra_vars['aux_y'], _ = compute_shifts(x=extra_vars['aux_data'], ctx=aux_ctx,
                extra_vars=extra_vars, default_parameters=parameters)
        oy, run_time = compute_shifts(x=x, ctx=ctx, 
            extra_vars=extra_vars, default_parameters=parameters) 
        y, aux_data = model_utils.data_postprocessing(x, oy, extra_vars)
        import ipdb;ipdb.set_trace()
        it_score = get_fits(y, gt, extra_vars)

    #Add to database
    update_data(parameters,extra_vars['figure_name'],parameters._id,it_score)
    return