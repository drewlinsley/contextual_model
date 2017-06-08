#!/usr/bin/env python

import time
import hmax.tools.utils as utils
import itertools as it
from copy import deepcopy
import numpy as np
from numpy import r_, s_
import scipy as sp
from scipy import stats
from scipy import ndimage
from scipy import special
from scipy.ndimage.filters import uniform_filter1d as u1d
from scipy.ndimage.filters import maximum_filter1d as m1d
from scipy.ndimage.filters import gaussian_filter1d as g1d
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from hmax.tools.utils import pb, mul, ifloor, iceil, iround
from hmax.models.hnorm.computation import pool, hwrectify
from hmax.models.ucircuits.contextual import stimuli as stim
# from hmax.models.ucircuits.contextual import contextual as cx
from hmax.models.ucircuits.contextual.contextual_CUDA import _DEFAULT_PARAMETERS
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import lines
from matplotlib import cm
# import prettyplotlib as ppl
import seaborn as sns
# import colormaps as custom_colormaps
# cm.register_cmap(name='viridis', cmap=custom_colormaps.viridis)
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['axes.grid'] = False
default_cmap = 'viridis'
plt.style.use('ggplot')

NOISE_LOC = 0.0
NOISE_SCALE = 0.0

DEFAULT_FDOMAIN = (0., 1.)#(-sp.pi, sp.pi)
DEFAULT_SCALE_GAUSSIAN = .1
DEFAULT_SCALE_CIRCULAR = 1.0/sp.sqrt(5.)
DEFAULT_SCALE_MONOTONIC = 25.
DEFAULT_SHAPE_LOGNORMAL = 0.20 # 1.00
DEFAULT_SCALE_LOGNORMAL = 0.75 # 0.20
DEFAULT_OFFSET_LOGNORMAL = 0.00 # 0.50
DEFAULT_BASELINE_AGAUSSIAN = (0., 0.5)
DEFAULT_SCALE_AGAUSSIAN = (.025, 0.5)

#------------------------------------------------------------------------------#
def sfit(ptx_eval, ptx, pty, s=1., k=3, t=None):
    """ Fit a bunch of points and return an evaluation vector """
    assert len(ptx) == len(pty)
    if t is None:
        return UnivariateSpline(x=ptx, y=pty, s=s, k=k)(ptx_eval)
    else:
        return LSQUnivariateSpline(x=ptx, y=pty, t=t, k=k)(ptx_eval)

def draw_ecrf(ax, loc=None, srf=_DEFAULT_PARAMETERS['srf'],
    ssn=_DEFAULT_PARAMETERS['ssn'], ssf=_DEFAULT_PARAMETERS['ssf'], off=0.5):

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    if loc is None:
        loc = [sum(xlims)/2., sum(ylims)/2.]

    ul = lambda sz: [loc[0]-ifloor(sz/2.0)-off, loc[1]-ifloor(sz/2.0)-off]
    ur = lambda sz: [loc[0]+iceil(sz/2.0)-off, loc[1]-ifloor(sz/2.0)-off]
    lr = lambda sz: [loc[0]+iceil(sz/2.0)-off, loc[1]+iceil(sz/2.0)-off]
    ll = lambda sz: [loc[0]-ifloor(sz/2.0)-off, loc[1]+iceil(sz/2.0)-off]

    # draw CRF
    ax.plot(ul(srf), ur(srf), color='r')
    ax.plot(ur(srf), lr(srf), color='r')
    ax.plot(lr(srf), ll(srf), color='r')
    ax.plot(ll(srf), ul(srf), color='r')

    # draw neCRF
    ax.plot(ul(ssn), ur(ssn), color='g')
    ax.plot(ur(ssn), lr(ssn), color='g')
    ax.plot(lr(ssn), ll(ssn), color='g')
    ax.plot(ll(ssn), ul(ssn), color='g')

    # draw feCRF
    ax.plot(ul(ssf), ur(ssf), color='b')
    ax.plot(ur(ssf), lr(ssf), color='b')
    ax.plot(lr(ssf), ll(ssf), color='b')
    ax.plot(ll(ssf), ul(ssf), color='b')

    # restore plot limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return

#------------------------------------------------------------------------------#
def nvonmisespdf(x, loc, scale):
    """ Normalized Von Mises density to be defined over [0, 1]
    rather than [-pi, pi]
    """

    return stats.vonmises.pdf(x*2*sp.pi-sp.pi,
        loc=loc*2*sp.pi-sp.pi, kappa=1.0/scale ** 2.0)

#------------------------------------------------------------------------------#
def bhattacharyya(arr1, arr2, axis=None):
    """"""

    arr1_sum = arr1.sum(axis=axis, keepdims=True)
    arr2_sum = arr2.sum(axis=axis, keepdims=True)

    return sp.sqrt(arr1/arr1_sum * arr2/arr2_sum).sum(axis=axis)

#------------------------------------------------------------------------------#
def agaussian(t, loc, bl, br, sl, sr, a=1.0):
    """ Using equation (5) from Salinas (2006) """

    yl = bl + (a - bl) * sp.exp(-(t - loc)**2 / sl**2)
    yr = br + (a - br) * sp.exp(-(t - loc)**2 / sr**2)

    return yl * (t <= loc) + yr * (t > loc)

#------------------------------------------------------------------------------#
def decode(arr, kind='vote', axis=None, fdomain=DEFAULT_FDOMAIN,
    exclude_percent_of_max=0., q=4):
    """"""

    if axis is None:
        if arr.ndim >=3:
            axis = -3
        elif arr.ndim == 1:
            axis = 0
        else:
            raise ValueError('Ambiguous axis along which to decode')

    nunits = arr.shape[axis]
    values = sp.linspace(fdomain[0], fdomain[1], nunits, dtype=float)

    if exclude_percent_of_max > 0:
        pct = arr.max(axis=axis, keepdims=True) * \
            vote_exclude_percent_of_max
        arr = sp.maximum(arr - pct, 0)

    if kind == 'argmax':
        return arr.argmax(axis=axis)/float(nunits-1) \
            * (fdomain[1] - fdomain[0]) + fdomain[0]

    elif kind == 'vote':
        return sp.tensordot(arr, values, ((axis,), (0,)))/arr.sum(axis)

    elif kind == 'voteExp':
        return sp.tensordot(arr**q, values, ((axis,), (0,)))/(arr**q).sum(axis)

    elif kind == 'circular_vote':
        values = (values - fdomain[0])/(fdomain[1] - fdomain[0])
        sv = sp.sin(values * 2*sp.pi - sp.pi)
        cv = sp.cos(values * 2*sp.pi - sp.pi)
        sw = sp.tensordot(arr, sv, ((axis,), (0,)))/arr.sum(axis)
        cw = sp.tensordot(arr, cv, ((axis,), (0,)))/arr.sum(axis)
        return (sp.arctan2(sw, cw) + sp.pi)/(2*sp.pi) \
            * (fdomain[1] - fdomain[0]) + fdomain[0]

    else:
        raise ValueError('Invalid decoder type')

#------------------------------------------------------------------------------#
def decode_bayes(arr, axis=None, density=None, scale=None):
    """ The density is the probabilistic equivalent of the tuning curve, with
    two input arguments: the first is the value and the second the location
    parameter of the density. Assumin fdomain is (0., 1.)
    """

    nunits = arr.shape[axis]
    values = sp.linspace(0.0, 1.0, nunits, dtype=float)

    # matrix of discrete tuning curves; i: stimulus value #ID, j: preference
    tuning = sp.array([[density(theta, loc=phi, scale=scale) \
        for theta in values] for phi in values])
    tuning /= tuning.sum(1, keepdims=True)

    # normalized spike count
    spksum = arr.sum(axis, keepdims=True)
    spksum[spksum == 0] = 1.0
    spikes = arr / spksum

    # now decode
    probas = sp.tensordot(spikes, tuning, ((axis,), (0,)))
    probas = sp.rollaxis(probas, axis=-1, start=axis)
    mapest = values[probas.argmax(axis)]
    #mapest = sp.tensordot(probas, values, ((axis,), (0,)))
    spread = 1.0 / (probas.max(axis) - probas.min(axis))

    return probas, mapest, spread

#-----------------------------------------------------------------------------#
def allow_multiple_stimulus_values(get_population):
    """ Decorator to allow for multiple stimulus values,
    i.e., multimodal LN population responses. 
    """

    def _get_population(*args, **kwargs):
        try:
            stimulus = kwargs.pop('stimulus')
        except KeyError:
            args = [arg for arg in args]
            stimulus = args.pop(0)
        if stimulus.ndim == 2:
            stimulus = stimulus.reshape((1, 1) + stimulus.shape)
        elif stimulus.ndim == 3:
            stimulus = stimulus.reshape((1,) + stimulus.shape)
        elif stimulus.ndim == 4:
            pass
        else:
            raise ValueError('Input stimulus has too many axes.')
        return sp.array([[get_population(subim, *args, **kwargs) \
            for subim in im] for im in stimulus]).max(1)

    return _get_population

#------------------------------------------------------------------------------#
def get_population2(*args, **kwargs):
    """ For now I don't want to adapt experiments to new format """
    return allow_multiple_stimulus_values(get_population)(*args, **kwargs)

#------------------------------------------------------------------------------#
# @allow_multiple_stimulus_values # TODO: adapt experiments to new format!!!
def get_population(stimulus, kind, noise_loc=NOISE_LOC, noise_scale=NOISE_SCALE,
    **kwargs):
    """"""

    if kind in ['gaussian', 'normal']:
        population = get_population_gaussian(stimulus, **kwargs)
    elif kind == 'circular':
        population = get_population_circular(stimulus, **kwargs)
    elif kind == 'monotonic':
        population = get_population_monotonic(stimulus, **kwargs)
    elif kind == 'lognormal':
        population = get_population_lognormal(stimulus, **kwargs)
    elif kind == 'agaussian':
        population = get_population_agaussian(stimulus, **kwargs)
    else:
        raise ValueError('Invalid distribution type for the tuning curves')

    if noise_scale:
        noise = sp.random.normal(loc=noise_loc,
            scale=noise_scale, size=population.shape)
        population += sp.maximum(noise, 0)

    bg_values = sp.isnan(population).any(axis=0)
    population[:, bg_values] = 0
    population /= (population.max() + (population.max() == 0))

    return population

#------------------------------------------------------------------------------#
def get_population_lognormal(stimulus, npoints=50, fdomain=DEFAULT_FDOMAIN,
    shape=DEFAULT_SHAPE_LOGNORMAL, scale=DEFAULT_SCALE_LOGNORMAL,
    offset=DEFAULT_OFFSET_LOGNORMAL):
    """"""

    # Create 'npoints' regularly-spaced tuning curves in the desired range 'fdomain'
    z, (h, w) = sp.linspace(fdomain[0], fdomain[1], npoints), stimulus.shape
    population = sp.zeros((npoints, h, w))

    for preferred, sl in zip(z, population):
        sl[:] = stats.lognorm.pdf(stimulus,
            s=shape, loc=preferred - offset, scale=scale)

    return population

#------------------------------------------------------------------------------#
def get_population_gaussian(stimulus, npoints=50, fdomain=DEFAULT_FDOMAIN,
    scale=DEFAULT_SCALE_GAUSSIAN):
    """"""

    # Create 'npoints' regularly-spaced tuning curves in the desired range 'fdomain'
    z, (h, w) = sp.linspace(fdomain[0], fdomain[1], npoints), stimulus.shape
    population = sp.zeros((npoints, h, w))

    for preferred, sl in zip(z, population):
        sl[:] = stats.norm.pdf(stimulus, loc=preferred, scale=scale)

    return population

#------------------------------------------------------------------------------#
def get_population_circular(stimulus, npoints=50, fdomain=DEFAULT_FDOMAIN,
    scale=DEFAULT_SCALE_CIRCULAR):
    """"""

    unit2ang = lambda a: (a - fdomain[0]) \
        /(fdomain[1]-fdomain[0]) * 2 * sp.pi - sp.pi

    # Create 'npoints' regularly-spaced tuning curves in the desired range 'fdomain'
    z, (h, w) = sp.linspace(fdomain[0], fdomain[1], npoints), stimulus.shape
    population = sp.zeros((npoints, h, w))

    z = unit2ang(z)
    s = unit2ang(stimulus)

    for preferred, sl in zip(z, population):
        kappa = 1.0 / scale ** 2
        sl[:] = stats.vonmises.pdf(s, loc=preferred, kappa=kappa)

    return population

#------------------------------------------------------------------------------#
def get_population_monotonic(stimulus, npoints=50, fdomain=DEFAULT_FDOMAIN,
    scale=DEFAULT_SCALE_MONOTONIC):
    """"""

    h, w = stimulus.shape
    alphas = sp.linspace(fdomain[0], fdomain[1], npoints)
    alphas = alphas.reshape((npoints, 1, 1))

    population = special.expit(scale/(fdomain[1] - fdomain[0]) * \
        (stimulus.reshape((1, h, w)) - alphas))

    return population

#------------------------------------------------------------------------------#
def get_population_agaussian(stimulus, npoints=50, fdomain=DEFAULT_FDOMAIN,
    baseline=DEFAULT_BASELINE_AGAUSSIAN, scale=DEFAULT_SCALE_AGAUSSIAN):
    """"""

    loci = np.linspace(fdomain[0], fdomain[1], npoints)
    b_min, b_max = min(baseline), max(baseline)
    s_min, s_max = min(scale), max(scale)

    bs_ = np.linspace(b_min, b_max, iceil(npoints/2.))
    _bs = np.linspace(b_max, b_min, iceil(npoints/2.))
    baselines = np.concatenate((_bs[:ifloor(npoints/2.)], bs_))

    sc_ = np.linspace(s_min, s_max, iceil(npoints/2.))
    _sc = np.linspace(s_max, s_min, iceil(npoints/2.))
    scales = np.concatenate((_sc[:ifloor(npoints/2.)], sc_))

    h, w = stimulus.shape
    population = np.zeros((npoints, h, w))

    for idx, (loc, bs, sc) in enumerate(zip(loci, baselines, scales)):
        if loc <= (fdomain[1] + fdomain[0])/2.:
            population[idx] = agaussian(stimulus, 
                loc=loc, bl=bs, br=b_min, sl=sc, sr=s_min)
        if loc > (fdomain[1] + fdomain[0])/2.:
            population[idx] = agaussian(stimulus,
                loc=loc, bl=b_min, br=bs, sl=s_min, sr=sc)

    return population

#------------------------------------------------------------------------------#
def show_population_static(x, idx_x, idx_y, fdomain=DEFAULT_FDOMAIN, kind='cartesian',
    ax=None, label=None, plot_kwargs={}, title=''):
    """ Docstring for show_population_static
    """
    
    if kind == 'cartesian':
        subplot_kw = {'polar': False}
    elif kind == 'polar':
        subplot_kw = {'polar': True}
    else:
        raise ValueError('Invalid graph type')

    if ax is None:
        fig, ax = ppl.plt.subplots(1, 1, subplot_kw=subplot_kw)
    else:
        fig = ppl.plt.gcf()

    for i, (idx, idy) in enumerate(zip(idx_x, idx_y)):
        line, = ppl.plot(ax, sp.linspace(fdomain[0], fdomain[1], len(x)),
            x[:, idx, idy], linewidth=2, label=label, **plot_kwargs)

    ax.set_xlim(fdomain)
    ax.set_title(title)

    return fig, ax, line

#------------------------------------------------------------------------------#
def show_population_animation_bars(X, idx_x=50, idx_y=50, interval=150,
    colormap=None, blit=False, kind='cartesian', save=None, show_initial=True,
    xlabel=u"\u03B8", ylabel=u"Y[\u03B8]", decoding_method='vote',
    fdomain=DEFAULT_FDOMAIN, frame_id=None, ax=None, show_legend=True,
    show_title=True):
    """ Docstring for show_population_animation
    """

    # WARNING: FOR THIS ANIMATION TO WORK,
    # THE OBJECT ANIM HAS TO BE RETURNED BY
    # THE CALLING FUNCTION.

    ################################################################
    # setup plot and sampli loci
    ################################################################
    try:
        import prettyplotlib as ppl
        plt = ppl.plt
        plot, legend = ppl.plot, ppl.legend
    except ImportError:
        import matplotlib.pyplot as plt
        plot, legend = plt.plot, plt.legend
    import matplotlib.animation as animation

    if kind == 'cartesian':
        subplot_kw = {'polar': False}
    elif kind == 'polar':
        subplot_kw = {'polar': True}
    else:
        raise ValueError('Invalid graph type')

    # Multiple sampling loci are allowed
    try:
        [idx for idx in idx_x]
    except TypeError:
        idx_x = [idx_x]
    try:
        [idx for idx in idx_y]
    except TypeError:
        idx_y = [idx_y]
    try:
        assert len(idx_x) == len(idx_y)
    except AssertionError:
        raise Exception('Coordinate lists idx_x and idx_y have different lengths')
    else:
        n_curves = len(idx_x)
    
    n_timesteps = X.shape[0]
    n_neurons = X.shape[1]
    xs = sp.arange(n_neurons)

    ################################################################
    # generate colors for histograms & other graphics parameters
    ################################################################
    if colormap is None:
        colormap = [plt.cm.gist_rainbow(k) \
            for k in sp.linspace(0, 0.9, n_curves)]
        # colormap = sns.color_palette(n_colors=n_curves)
        # colormap = sns.color_palette("husl", n_curves)

    if show_initial:
        INITIAL_LINE_WIDTH = 5.
    else:
        INITIAL_LINE_WIDTH = 0.

    if n_curves < 4:
        ALPHA_BAR_NORMAL = .33
        ALPHA_BAR_HIGHLIGHTED = .60
    else:
        ALPHA_BAR_NORMAL = .20
        ALPHA_BAR_HIGHLIGHTED = .75
    BAR_WIDTH = .75

    ################################################################
    # create main figure
    ################################################################
    if not ax:
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
    else:
        fig = plt.gcf()
    # ax.patch.set_facecolor('white')
    y = sp.array([X[:, :, i_, j_] for i_, j_ in zip(idx_x, idx_y)])
    for idx, (i_, j_) in enumerate(zip(idx_x, idx_y)):
        y[idx, :, :] = X[:, :, i_, j_]
    ax.set_xlim([0, n_neurons-1])
    ax.set_ylim([0, y.max()])

    xticks_val = sp.linspace(fdomain[0], fdomain[1], 21)
    xticks_pos = sp.linspace(0, n_neurons-1, 21) + BAR_WIDTH/2.
    ax.set_xticks(xticks_pos, minor=False)
    ax.set_xticklabels(['%1.2f' % i for i in xticks_val], minor=False)

    plt.show(); plt.draw()

    ################################################################
    # define decoder from population to index
    ################################################################
    def decode_index(y_idx_i, kind=decoding_method):
        """ get population-index of decoded value """

        decoded_value_index = int(sp.around(decode(y_idx_i,
            kind=kind, axis=0) * n_neurons))

        # corner case if vote decoding over circular variable
        decoded_value_index = 0 if decoded_value_index \
            == n_neurons else decoded_value_index

        return decoded_value_index

    ################################################################
    # draw initial population curves at time = 0
    ################################################################
    grpbars, rline = [], []
    for idx, (i_, j_) in enumerate(zip(idx_x, idx_y)):
        bars = ppl.bar(
            xs, y[idx][0],
            ax=ax, width=BAR_WIDTH, color=colormap[idx],
            alpha=ALPHA_BAR_NORMAL, linewidth=0., edgecolor='k',
            label='Normalized population response')
        grpbars.append(bars)
    grpbars = tuple(grpbars)

    # set alpha for bar-of-decoded, initial frame
    for idx in sp.arange(n_curves):
        decoded_value_index = decode_index(y[idx][0])
        for j, b in enumerate(grpbars[idx]):
            if j == decoded_value_index:
                b.set_linewidth(0)
                b.set_alpha(ALPHA_BAR_HIGHLIGHTED)

    ################################################################
    # Draw vertical lines at loci of population curves maxima, or,
    # in the polar case, radial lines from center to loci of
    # population curves maxima
    ################################################################
    for idx in range(n_curves):
        max_y0 = y[idx][0].max()
        decoded_value_index = decode_index(y[idx][0])

        rline_idx = ax.add_line(plt.Line2D(
            xdata=(decoded_value_index + BAR_WIDTH/2.,
                decoded_value_index + BAR_WIDTH/2.),
            ydata=ax.get_ylim(),
            color=colormap[idx],
            linestyle='--',
            alpha=.50,
            linewidth=INITIAL_LINE_WIDTH))

        rline.append(rline_idx)
    rline = tuple(rline)

    ################################################################
    # Axes, labels and legends
    ################################################################
    if kind == 'cartesian':
        ax.set_xlabel(xlabel, labelpad=15., fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        title_ = ax.set_title('')
        ax.tick_params(axis='both', which='major')
        ax.xaxis.set_label_position('bottom')
    elif kind == 'polar':
        pass

    if show_legend:
        legend_patches = [patches.Patch(
            color=c, label='Population at [%i, %i]' \
            % (i, j)) for i, j, c in zip(idx_x, idx_y, colormap)]
        legend_lines = [lines.Line2D([0, 0], [0, 0],
            linewidth=2*INITIAL_LINE_WIDTH, linestyle='--',
            color=c, label='Initial decoding at [%i, %i]' % (i, j)) \
            for i, j, c in zip(idx_x, idx_y, colormap)]
        legend_elements = legend_patches + legend_lines
        ax.legend(handles=legend_elements)
        plt.show(); plt.draw()

    ################################################################
    # Animate: this function *must* return one tuple for all changed
    # objects for animation.FuncAnimation to know what it must
    # update in the figure
    ################################################################
    def update_plot(i):
        if show_title:
            title_.set_text('Iteration: t = %i / %i' % (i+1, len(X)))
            fig.canvas.set_window_title('t = %i / %i' % (i+1, len(X)))
        if i == -1:
            return grpbars
        for idx in sp.arange(n_curves):
            decoded_value_index = decode_index(y[idx][i])
            for j, b in enumerate(grpbars[idx]):
                b.set_height(y[idx][i][j])
                b.set_linewidth(0)
                b.set_alpha(ALPHA_BAR_NORMAL)
                if j == decoded_value_index:
                    b.set_linewidth(0)
                    b.set_alpha(ALPHA_BAR_HIGHLIGHTED)
        plt.show(); plt.draw()

        return grpbars

    ################################################################
    # initialization function
    ################################################################
    def init(): return update_plot(0)

    ################################################################
    # run animation
    ################################################################
    try:
        return update_plot(frame_id-1)
    except AssertionError:
        raise ValueError('Frame ID should be positive')
    except TypeError:
        raw_input("Press Enter to start animation.")

        anim = animation.FuncAnimation(fig, update_plot, len(X),
            init_func=init, interval=interval, blit=blit)
        plt.show(); plt.draw()

        if save:
            anim.save(save, writer='avconv')

        return anim

#------------------------------------------------------------------------------#
def animate_populations(X, idx,
    curve_colors=None,
    curve_labels=None,
    curve_type='bar',
    interval=150,
    blit=False,
    kind='cartesian',
    save=None,
    show_initial=True,
    xlabel=u"\u03B8",
    ylabel=u"Y[\u03B8]",
    decoding_method='vote',
    fdomain=DEFAULT_FDOMAIN,
    frame_id=None,
    ax=None,
    show_legend=True,
    show_title=True):
    """ X ~ (t, n, k, h, w); idx is a list of 3-tuples (i_n, i_h, i_w)
    """

    # WARNING: FOR THIS ANIMATION TO WORK,
    # THE OBJECT ANIM HAS TO BE RETURNED BY
    # THE CALLING FUNCTION.

    ################################################################
    # setup plot and sampli loci
    ################################################################
    assert curve_type in ('bar', 'line')
    try:
        import prettyplotlib as ppl
        plt = ppl.plt
        plot, legend = ppl.plot, ppl.legend
    except ImportError:
        import matplotlib.pyplot as plt
        plot, legend = plt.plot, plt.legend
    import matplotlib.animation as animation

    if kind == 'cartesian':
        subplot_kw = {'polar': False}
    elif kind == 'polar':
        subplot_kw = {'polar': True}
    else:
        raise ValueError('Invalid graph type')

    # Multiple sampling loci are allowed
    n_curves = len(idx)
    n_timesteps = X.shape[0]
    n_arrays = X.shape[1]
    n_neurons = X.shape[2]
    xs = sp.arange(n_neurons)
    for electrode in idx:
        assert len(electrode) == 3

    ################################################################
    # generate colors for histograms & other graphics parameters
    ################################################################
    if curve_colors is None:
        curve_colors = [plt.cm.gist_rainbow(k) \
            for k in sp.linspace(0, 0.9, n_curves)]
    else:
        assert len(curve_colors) == n_curves

    if curve_labels is None:
        curve_labels = ['%i' % (i,) for i in range(n_curves)]
    else:
        assert len(curve_labels) == n_curves

    if show_initial:
        INITIAL_LINE_WIDTH = 5.
    else:
        INITIAL_LINE_WIDTH = 0.

    if n_curves < 4:
        ALPHA_BAR_NORMAL = .33
        ALPHA_BAR_HIGHLIGHTED = .60
    else:
        ALPHA_BAR_NORMAL = .20
        ALPHA_BAR_HIGHLIGHTED = .75
    BAR_WIDTH = .75

    ################################################################
    # create main figure
    ################################################################
    if not ax:
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
    else:
        fig = plt.gcf()
    # ax.patch.set_facecolor('white')
    X_curves = sp.array([X[:, n_, :, i_, j_] for (n_, i_, j_) in idx])
    assert X_curves.shape == (n_curves, n_timesteps, n_neurons)

    ax.set_xlim([0, n_neurons-1])
    ax.set_ylim([0, X_curves.max()])

    xticks_val = sp.linspace(fdomain[0], fdomain[1], 21)
    xticks_pos = sp.linspace(0, n_neurons-1, 21) + BAR_WIDTH/2.
    ax.set_xticks(xticks_pos, minor=False)
    ax.set_xticklabels(['%1.2f' % i for i in xticks_val], minor=False)

    plt.show(); plt.draw()

    ################################################################
    # define decoder from population to index
    ################################################################
    def decode_index(y_idx_i, kind=decoding_method):
        """ get population-index of decoded value """
        try:
            decoded_value_index = int(sp.around(decode(y_idx_i,
                kind=kind, axis=0) * n_neurons))
        except ValueError:
            decoded_value_index = 0

        # corner case if vote decoding over circular variable
        decoded_value_index = 0 if decoded_value_index \
            == n_neurons else decoded_value_index

        return decoded_value_index

    ################################################################
    # draw initial population curves at time = 0
    ################################################################
    population_graphs = []
    for i_curve in range(n_curves):
        if curve_type == 'bar':
            obj = ax.bar(xs, X_curves[i_curve][0],
                width=BAR_WIDTH,
                color=curve_colors[i_curve],
                alpha=ALPHA_BAR_NORMAL,
                linewidth=0.,
                edgecolor='k',
                label='Normalized population response')
        else:
            obj, = ax.plot(xs, X_curves[i_curve][0],
                color=curve_colors[i_curve],
                alpha=ALPHA_BAR_NORMAL,
                linewidth=3.,
                label='Normalized population response')
            obj = (obj,)
        population_graphs.append(obj)
    population_graphs = tuple(population_graphs)

    # set alpha for bar-of-decoded, initial frame
    for i_curve in range(n_curves):
        max_y0 = X_curves[i_curve][0].max()
        min_y0 = X_curves[i_curve][0].min()
        decoded_value_index = decode_index(X_curves[i_curve][0])

        # if population amplitude is < 1% of max value, then it does
        # not make sense to try and decode an initial value.
        if ((max_y0 - min_y0) > max_y0/100.) and (curve_type == 'bar'):
            for j, b in enumerate(population_graphs[i_curve]):
                if j == decoded_value_index:
                    b.set_linewidth(0)
                    b.set_alpha(ALPHA_BAR_HIGHLIGHTED)

    ################################################################
    # Draw vertical lines at loci of population curves maxima, or,
    # in the polar case, radial lines from center to loci of
    # population curves maxima
    ################################################################
    rline = []
    for i_curve in range(n_curves):
        max_y0 = X_curves[i_curve][0].max()
        min_y0 = X_curves[i_curve][0].min()
        decoded_value_index = decode_index(X_curves[i_curve][0])

        # if population amplitude is < 1% of max value, then it does
        # not make sense to try and decode an initial value.
        if (max_y0 - min_y0) > max_y0/100.:
            rline_idx = ax.add_line(plt.Line2D(
                xdata=(decoded_value_index + BAR_WIDTH/2.,
                    decoded_value_index + BAR_WIDTH/2.),
                ydata=ax.get_ylim(),
                color=curve_colors[i_curve],
                linestyle='--',
                alpha=.50,
                linewidth=INITIAL_LINE_WIDTH))
            rline.append(rline_idx)
    rline = tuple(rline)

    ################################################################
    # Axes, labels and legends
    ################################################################
    if kind == 'cartesian':
        ax.set_xlabel(xlabel, labelpad=15., fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        title_ = ax.set_title('')
        ax.tick_params(axis='both', which='major')
        ax.xaxis.set_label_position('bottom')
    elif kind == 'polar':
        pass

    if show_legend:
        legend_patches = [patches.Patch(
            color=c, label='Population at [%s: %i, %i]' % (l, i, j)) \
            for (_, i, j), c, l in zip(idx, curve_colors, curve_labels)]

        legend_lines = [lines.Line2D([0, 0], [0, 0],
            linewidth=2*INITIAL_LINE_WIDTH, linestyle='--',
            color=c, label='Initial decoding at [%s: %i, %i]' % (l, i, j)) \
            for (_, i, j), c, l in zip(idx, curve_colors, curve_labels)]

        legend_elements = legend_patches + legend_lines
        ax.legend(handles=legend_elements)
        plt.show(); plt.draw()

    ################################################################
    # Animate: this function *must* return one tuple for all changed
    # objects for animation.FuncAnimation to know what it must
    # update in the figure
    ################################################################
    def update_plot(i):
        if show_title:
            title_.set_text('Iteration: t = %i / %i' % (i+1, len(X)))
            fig.canvas.set_window_title('t = %i / %i' % (i+1, len(X)))
        if i == -1:
            return population_graphs
        for i_curve in range(n_curves):
            max_yt = X_curves[i_curve][i].max()
            min_yt = X_curves[i_curve][i].min()
            decoded_value_index = decode_index(X_curves[i_curve][i])
            
            if curve_type == 'bar':
                for j, b in enumerate(population_graphs[i_curve]):
                    b.set_height(X_curves[i_curve][i][j])
                    b.set_linewidth(0)
                    b.set_alpha(ALPHA_BAR_NORMAL)

                    if (max_yt - min_yt) > max_yt/100.:
                        if j == decoded_value_index:
                            b.set_linewidth(0)
                            b.set_alpha(ALPHA_BAR_HIGHLIGHTED)
            else:
                for j, l in enumerate(population_graphs[i_curve]):
                    l.set_ydata(X_curves[i_curve][i])
        plt.show()
        plt.draw()

        return population_graphs

    ################################################################
    # initialization function
    ################################################################
    def init(): return update_plot(0)

    ################################################################
    # run animation
    ################################################################
    try:
        return update_plot(frame_id-1)
    except AssertionError:
        raise ValueError('Frame ID should be positive')
    except TypeError:
        raw_input("Press Enter to start animation.")

        anim = animation.FuncAnimation(fig, update_plot, n_timesteps,
            init_func=init, interval=interval, blit=blit)
        plt.show(); plt.draw()

        if save:
            anim.save(save, writer='avconv')

        return anim

#------------------------------------------------------------------------------#

def data_postprocessing(x,y,extra_vars):
    if extra_vars['figure_name'] == 'f3':
        cpt = extra_vars['cpt']
        spt = extra_vars['spt']
        cval = extra_vars['cval']
        sval = extra_vars['sval']

        # compute absolute and relative dshifts
        xdec = decode(x[:, cpt[0], cpt[1], :], axis=1, kind=extra_vars['decoder'])
        cdec = decode(y[:, cpt[0], cpt[1], :], axis=1, kind=extra_vars['decoder'])
        sdec = decode(y[:, spt[0], spt[1], :], axis=1, kind=extra_vars['decoder'])
        dshift_a = cdec - xdec
        dshift_r = (cdec - sdec)/(cval - sval)
        return dshift_a, dshift_r, run_time
    elif extra_vars['figure_name'] == 'f4':
        pass
    elif extra_vars['figure_name'] == 'f5':
        import ipdb;ipdb.set_trace()
        a = 2
