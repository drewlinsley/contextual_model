import sys
import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
from timeit import default_timer as timer
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
sys.path.append('/home/drew/Documents/')
import model_utils
from model_defs.model_cpu_port_scan_optim import ContextualCircuit
from hmax.tools.utils import iceil


def adjust_parameters(defaults, hps):
    hps_keys = hps.keys()
    for k in defaults._DEFAULT_PARAMETERS.iteritems():
        if k in hps_keys:
            defaults._DEFAULT_PARAMETERS[k] = hps[k]
    return defaults


def printProgress(
        iteration, total, prefix='', suffix='',
        decimals=1, bar_length=100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = ' ' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(
        '\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def compute_shifts(x, sess, ctx, extra_vars, default_parameters):
    start = timer()
    sess.run(
        tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()))
    # tf.group(tf.global_variables_initializer())
    feed_dict = {
        ctx.X: x,
        ctx.alpha: np.asarray(
            default_parameters._DEFAULT_PARAMETERS['alpha']).reshape(-1, 1),
        ctx.beta: np.asarray(
            default_parameters._DEFAULT_PARAMETERS['beta']).reshape(-1, 1),
        ctx.mu: np.asarray(
            default_parameters._DEFAULT_PARAMETERS['mu']).reshape(-1, 1),
        ctx.nu: np.asarray(
            default_parameters._DEFAULT_PARAMETERS['nu']).reshape(-1, 1),
        ctx.gamma: np.asarray(
            default_parameters._DEFAULT_PARAMETERS['gamma']).reshape(-1, 1),
        ctx.delta: np.asarray(
            default_parameters._DEFAULT_PARAMETERS['delta']).reshape(-1, 1)
    }
    # ctx.xi:default_parameters._DEFAULT_PARAMETERS['xi'].reshape(-1,1),
    if extra_vars['return_var'] == 'I':
        y = sess.run(ctx.out_I, feed_dict=feed_dict)
    elif extra_vars['return_var'] == 'O':
        y = sess.run(ctx.out_O, feed_dict=feed_dict)
    end = timer()
    run_time = end - start
    return y, run_time


def prepare_hps(parameters, hp_set):
    new_parameters = deepcopy(parameters)
    for key in parameters.tunable_params:
        new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(hp_set[key])
    return new_parameters


def produce_plots(y, lesion, extra_vars, parameters):
    if extra_vars['figure_name'] == 'size_tuning':
        fig, ax = plt.subplots()
        y = y.reshape(
            len(extra_vars['contrasts']),
            y.shape[0]//len(extra_vars['contrasts']),
            y.shape[1], y.shape[2], y.shape[3])

        # size tuning curve
        for idx, cst in enumerate(extra_vars['contrasts']):
            ax.plot(
                extra_vars['stimsizes'],
                y[
                    idx, :, extra_vars['npoints']//2, extra_vars['size']//2,
                    extra_vars['size']//2],
                color=extra_vars['curvecols'][idx], linewidth=5., alpha=.50)
            ax.plot(
                extra_vars['stimsizes'],
                y[
                    idx, :, extra_vars['npoints']//2, extra_vars['size']//2,
                    extra_vars['size']//2],
                color=extra_vars['curvecols'][idx],
                label=extra_vars['curvelabs'][idx],
                marker='o', linestyle='none', alpha=.75)
        # misc
        ax.plot(
            [1, 1], ax.get_ylim(), linewidth=3., linestyle='--', alpha=.75,
            color='#FF1900', label='Classical receptive field')
        ax.plot(
            [extra_vars['ssn'], extra_vars['ssn']], ax.get_ylim(),
            linewidth=3., linestyle='--', alpha=.75,
            color='#3FA128', label='Near surround')
        ax.plot(
            [extra_vars['ssf'], extra_vars['ssf']], ax.get_ylim(),
            linewidth=3., linestyle='--', alpha=.75,
            color='#006AFF', label='Far surround')
        ax.y_lim([0, 12])
        ax.set_xlabel('Stimulus size', fontweight='bold', fontsize=15.)
        ax.set_ylabel('Cell response', fontweight='bold', fontsize=15.)

        plt.title('%s with lesion %s' % (extra_vars['figure_name'], lesion))
        plt.savefig(
            os.path.join(
                parameters._FIGURES,
                '%s_%s.pdf') % (lesion, extra_vars['figure_name']))
    else:
        pass


def get_fits(y, gt, extra_vars):
    if extra_vars['figure_name'] == 'tbp':
        tcs = np.zeros((2, len(extra_vars['vals'])))
        for i in range(len(extra_vars['vals'])):
            tcs[0, i] = np.corrcoef(
                y[0, i][::iceil(
                    extra_vars['npoints']/float(
                        len(extra_vars['vals'])))], gt[0][i, 1:])[0, 1]
            tcs[1, i] = np.corrcoef(
                y[1, i][::iceil(extra_vars['npoints']/float(
                    len(extra_vars['vals'])))], gt[1][i, 1:])[0, 1]
        it_score = np.mean(tcs)  # ** 2)
    elif extra_vars['figure_name'] == 'tbtcso':
        tcs = np.zeros((y.shape[0]))
        for i in range(y.shape[0]):
            tcs[i] = np.corrcoef(y[i, :], gt[0][i, :])[0, 1]
        it_score = np.mean(tcs)  # ** 2)
    elif extra_vars['figure_name'] == 'bw':
        tcs = np.zeros((y.shape[0], y.shape[1]))
        for r in range(y.shape[0]):
            for c in range(y.shape[1]):
                tcs[r, c] = np.corrcoef(
                    np.squeeze(
                        y[r, c, :]), np.squeeze(gt[1][r, c, :]))[0, 1] ** 2
        it_score = np.nanmean(tcs)
    elif extra_vars['figure_name'] == 'tbtcso':
        rs = np.zeros((gt.shape[0]))
        for i, r in enumerate(gt):
            rs[i] = np.corrcoef(y, r)
        it_score = np.mean(rs)
    else:
        it_score = np.corrcoef(y, gt)[0, 1]  # ** 2
    return it_score


def optimize_model(im, gt, extra_vars, parameters):

    # Do outer loop of lesions -- embed in variable scopes
    for lesion in parameters.lesions:

        # outer_parameters = deepcopy(parameters)
        if 'hp_file' in extra_vars.keys():
            hps = np.load(extra_vars['hp_file'])['max_row'].item()[lesion]
            it_parameters = prepare_hps(parameters, hps)
        else:
            it_parameters = deepcopy(parameters)
        # it_parameters = prepare_hps(outer_parameters, hps[lesion])
        it_parameters.lesions = lesion
        print(
            'Running optimizations on problem ' +
            extra_vars['figure_name'] + ' after lesioning: ' +
            ' '.join(it_parameters.lesions))

        # Optional stuff and data prep
        if extra_vars['figure_name'] == 'f4':  # precalculated
            x = im.transpose(0, 2, 3, 1)  # need to copy or what?
            with tf.device('/gpu:0'):
                with tf.variable_scope('aux_' + lesion):
                    aux_ctx = ContextualCircuit(lesions=lesion)
                    aux_ctx.run(extra_vars['aux_data'])
                    # builds tf graph with shape of x
        elif extra_vars['figure_name'] == 'tbp' or\
                extra_vars['figure_name'] == 'tbtcso' or\
                extra_vars['figure_name'] == 'bw' or\
                extra_vars['figure_name'] == 'cross_orientation_suppression' or\
                extra_vars['figure_name'] == 'size_tuning' or\
                extra_vars['figure_name'] == 'cnn_features':
            x = im.transpose(0, 2, 3, 1)
        else:
            x = model_utils.get_population2(
                im, npoints=extra_vars['npoints'], kind=extra_vars['kind'],
                scale=extra_vars['scale']).astype(  # transpose to bhwc
                it_parameters._DEFAULT_FLOATX_NP).transpose(0, 2, 3, 1)

        # Build main graph
        with tf.device('/gpu:0'):
            with tf.variable_scope('main_' + lesion):
                ctx = ContextualCircuit(lesions=it_parameters.lesions)
                ctx.run(x)  # builds tf graph with shape of x

        # Special case for fig 4
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if extra_vars['figure_name'] == 'f4':
                extra_vars['aux_y'], _ = compute_shifts(
                    x=extra_vars['aux_data'], sess=sess, ctx=aux_ctx,
                    extra_vars=extra_vars, default_parameters=it_parameters)
            oy, run_time = compute_shifts(
                x=x, sess=sess, ctx=ctx, extra_vars=extra_vars,
                default_parameters=it_parameters)

        y, aux_data = model_utils.data_postprocessing(x, oy, extra_vars)
        if 'save_file' in extra_vars.keys():
            np.save(extra_vars['save_file'], y)

        if gt is not None:
            it_score = get_fits(y, gt, extra_vars)
            return it_score
        else:
            produce_plots(y, it_parameters.lesions, extra_vars, parameters)
