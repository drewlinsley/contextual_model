import sys, os
import numpy as np
import scipy as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
sys.path.append('/home/drew/Documents/')
import tensorflow as tf
import model_utils
from tqdm import tqdm
from copy import deepcopy
from timeit import default_timer as timer
from model_defs.model_cpu_port_scan_optim import ContextualCircuit
from db_utils import update_data, gather_data
from fig_6_utils import get_rdots
from scipy.optimize import curve_fit

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

def logistic(t, beta0, beta1):
    alpha0 = 0.5 #extra_vars['value_up'] - extra_vars['value_down']
    alpha1 = 0.3 #extra_vars['value_down']
    return logistic_full(t, beta0, beta1, alpha0, alpha1)

def gaussian(t, alpha, mu, sigma):
    return alpha * sp.exp(-(t - mu)**2/sigma**2)

def fit_gaussian(x, y, x_eval):
    try:
        alpha, mu, sigma = curve_fit(gaussian, x, y,
            p0=[y.max(), x[y.argmax()], 0.5])[0]
    except RuntimeError:
        alpha, mu, sigma = [sp.nan, sp.nan, sp.nan]
    finally:
        y_eval = gaussian(x_eval, alpha, mu, sigma)
    return alpha, mu, sigma, y_eval

def logistic_full(t, beta0, beta1, alpha0, alpha1):
    return alpha0/(1.0 + sp.exp(beta0*(t - beta1))) + alpha1

def fit_logistic(x, y, x_eval):
    try:
        beta0, beta1 = curve_fit(logistic, x, y, p0=[5.0, 0.0])[0]
    except RuntimeError:
        beta0, beta1 = [sp.nan, sp.nan]
    finally:
        y_eval = logistic(x_eval, beta0, beta1)
    return beta0, beta1, y_eval

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

def compute_shifts_loop(x, ctx, extra_vars, default_parameters):
    y = []
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
        for idx in range(extra_vars['ntrials']):
            if extra_vars['preprocess'] != None:
                nx = extra_vars['preprocess'](x,extra_vars)
            feed_dict['X'] = nx
            if extra_vars['return_var'] == 'I':
                y = np.append(y,sess.run(ctx.out_I,feed_dict=feed_dict))
            elif extra_vars['return_var'] == 'O':
                y = np.append(y,sess.run(ctx.out_O,feed_dict=feed_dict))
        end = timer()
        run_time = end - start
    return y,x,run_time

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

def prepare_hps(parameters,hp_sets,idx):
    new_parameters = deepcopy(parameters)
    for key in parameters.tunable_params:
        new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(hp_sets[key][idx])
    return new_parameters

def optimize_model(gt,extra_vars,parameters):
    #Taking the MurakamiShimojo1996 code and integrating with the hp_optim

    # parameters
    ###########
    nw = len(extra_vars['ws'])
    abs_cohs = sp.linspace(0, 100, extra_vars['ncoh']//2+1)/100.
    dirs = sp.linspace(0, 9, extra_vars['ndirs'], dtype=int)/float(extra_vars['ndirs'])
    coh50 = (extra_vars['ndirs']/2.0 - 1.0)/(extra_vars['ndirs'] - 1.0)
    scoh = coh50

    coherences = sp.linspace(-100, 100, extra_vars['ncoh'])/100.
    coherences4fit = sp.linspace(-100, 100, extra_vars['ncoh4fit'])/100.

    print('Running hyperparameter optimizations after lesioning: ' + ' '.join(parameters.lesions))
    build_graph = True
    for ol, w in enumerate(extra_vars['ws']):
        # generate random dot stimuli and simulate circuit
        ##################################################
        dec_dir_vote = sp.zeros((extra_vars['ncoh'], extra_vars['ntrials']))
        dec_variance = sp.zeros((extra_vars['ncoh'], extra_vars['ntrials']))
        for il in range(extra_vars['ntrials']):
            # generate stimuli at various center coherences
            im_poscoh = [get_rdots(cval=extra_vars['value_down'], ccoh=ccoh,
                sval=extra_vars['value_down'], scoh=scoh, density=extra_vars['rdd'], size=extra_vars['size'], w=w, dirs=dirs, ndirs=extra_vars['ndirs']) for ccoh in abs_cohs]
            im_negcoh = [get_rdots(cval=extra_vars['value_up'], ccoh=ccoh,
                sval=extra_vars['value_down'], scoh=scoh, density=extra_vars['rdd'], size=extra_vars['size'], w=w, dirs=dirs, ndirs=extra_vars['ndirs']) for ccoh in abs_cohs]
            im = sp.array(im_negcoh[1:][::-1] + im_poscoh)
            im.shape = (extra_vars['ncoh'], 1, extra_vars['size'], extra_vars['size'])

            # get population
            x = model_utils.get_population2(im, kind=extra_vars['kind'],
                scale=extra_vars['scale'], npoints=extra_vars['nunits']).transpose(0,2,3,1) #transpose to bhwc
            if build_graph:
                with tf.device('/gpu:0'):
                    with tf.variable_scope('main'):
                        ctx = ContextualCircuit(lesions=parameters.lesions)
                        ctx.run(x) #builds tf graph with shape of x
                build_graph = False

            oy, run_time = compute_shifts(x=x, ctx=ctx, 
                extra_vars=extra_vars, default_parameters=parameters) 
            dec_dir_vote[:,il], dec_variance[:,il]= model_utils.data_postprocessing(x,oy,extra_vars) #maybe return a dict instead?

        PSE_vote = sp.zeros((nw,))
        PSE_variance = sp.zeros((nw,))
        dec_dir_vote_fit = sp.zeros((nw, extra_vars['ncoh4fit']))
        dec_variance_fit = sp.zeros((nw, extra_vars['ncoh4fit']))
        _, PSE_vote[ol], dec_dir_vote_fit[ol] = fit_logistic(
            x=coherences, y=dec_dir_vote[ol].mean(-1), x_eval=coherences4fit)
        _, _, _, dec_variance_fit[ol] = fit_gaussian(
            x=coherences, y=dec_variance[ol].mean(-1), x_eval=coherences4fit)
        PSE_variance[ol] = coherences4fit[dec_variance_fit[ol].argmax()]
    PSE_mixed = (PSE_vote + PSE_variance) / 2.0
    y = PSE_mixed * 100 #this is unnecessary for correlation, but let's keep for plotting purposes
    it_score = np.corrcoef(y,gt)[0,1] ** 2

    #Add to database
    import ipdb;ipdb.set_trace()
    update_data(parameters,extra_vars['figure_name'],parameters._id,it_score)
    return #scores,params
