import sys, os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
sys.path.append('/home/drew/Documents/')
import tensorflow as tf
import model_utils
from tqdm import tqdm
from copy import deepcopy
from hmax.tools.utils import iround
from timeit import default_timer as timer
from model_defs.model_cpu_port_scan_optim import ContextualCircuit

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
        ctx.alpha:default_parameters._DEFAULT_PARAMETERS['alpha'].reshape(-1,1),
        ctx.beta:default_parameters._DEFAULT_PARAMETERS['beta'].reshape(-1,1),
        ctx.mu:default_parameters._DEFAULT_PARAMETERS['mu'].reshape(-1,1),
        ctx.nu:default_parameters._DEFAULT_PARAMETERS['nu'].reshape(-1,1),
        ctx.gamma:default_parameters._DEFAULT_PARAMETERS['gamma'].reshape(-1,1),
        ctx.delta:default_parameters._DEFAULT_PARAMETERS['delta'].reshape(-1,1)}
        if extra_vars['return_var'] == 'I':
            y = sess.run(ctx.out_I,feed_dict=feed_dict)
        elif extra_vars['return_var'] == 'O':
            y = sess.run(ctx.out_O,feed_dict=feed_dict)
        end = timer()
        run_time = end - start    

    a,r = model_utils.data_postprocessing(x,y,extra_vars) #maybe return a dict instead?
    return a,y,run_time

def uniform_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] - ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] + ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        lin_range = np.linspace(low_lim,high_lim,parameters.tune_max_scale[idx])
        new_parameters._DEFAULT_PARAMETERS[key] = np.random.choice(lin_range)
    return new_parameters

def random_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] - ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] + ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        new_parameters._DEFAULT_PARAMETERS[key] = (high_lim - low_lim) * np.random.random() + low_lim
    return new_parameters

def random_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] - ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] + ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        new_parameters._DEFAULT_PARAMETERS[key] = (high_lim - low_lim) * np.random.random() + low_lim
    return new_parameters

def random_log_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] / 10
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] * 10
        new_parameters._DEFAULT_PARAMETERS[key] = (high_lim - low_lim) * np.random.random() + low_lim
    return new_parameters

def pass_params(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(new_parameters._DEFAULT_PARAMETERS[key])
    return new_parameters

def grab_sampler(parameters):
    if parameters.hp_optim_type == 'uniform':
        ofun = uniform_sampling
    elif parameters.hp_optim_type == 'random':
        ofun = random_sampling
    elif parameters.hp_optim_type == 'random_log':
        ofun = random_log_sampling
    elif parameters.hp_optim_type == 'none':
        ofun = pass_params
    return ofun

def optimize_model(im,gt,extra_vars,parameters):

    #Get data
    if extra_vars['figure_name'] == 'f4': #precalculated
        x = im #need to copy or what?
    else:
        x = model_utils.get_population2(im,
            npoints=extra_vars['npoints'], kind=extra_vars['kind'],
             scale=extra_vars['scale']).astype(parameters._DEFAULT_FLOATX_NP).transpose(0,2,3,1) #transpose to bhwc

    #Build graph
    with tf.device('/gpu:0'):
        ctx = ContextualCircuit(lesions=parameters.lesions)
        ctx.run(x) #builds tf graph with shape of x
    ofun = grab_sampler(parameters)
    scores = []
    params = []
    print('Running hyperparameter optimizations after lesioning: ' + ' '.join(parameters.lesions))

    for idx in range(parameters.iterations):
        random_parameters = ofun(parameters)
        y, aux_vars, run_time = compute_shifts(x=x, ctx=ctx, 
            extra_vars=extra_vars, default_parameters=random_parameters) 
        it_score = np.corrcoef(y,gt)[0,1] ** 2
        scores = np.append(scores,it_score)
        params.append(random_parameters._DEFAULT_PARAMETERS)
        printProgress(idx, parameters.iterations, 
            prefix = 'Progress:', 
            suffix = 'Iteration time: ' + str(np.around(run_time,2)) + '; Rsquared: ' + str(np.around(it_score,2)), 
            bar_length = 30)
    print('\n')
    return scores,params
