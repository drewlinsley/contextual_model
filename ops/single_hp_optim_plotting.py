import os
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def plot_y_history(y_hist):
    f = plt.figure()
    plt.plot(y_hist)
    plt.show()
    plt.close(f)


def produce_plots(y, lesion, extra_vars, parameters, max_channels=True):
    if extra_vars['figure_name'] == 'size_tuning':
        fig, ax = plt.subplots()
        y = y.reshape(
            len(extra_vars['contrasts']),
            y.shape[0]//len(extra_vars['contrasts']),
            y.shape[1], y.shape[2], y.shape[3])

        # size tuning curve
        for idx, cst in enumerate(extra_vars['contrasts']):
            if max_channels:
                it_y = y[
                    idx, :, extra_vars['npoints']//2, :, :]
                it_y = np.max(np.max(it_y, axis=-1), axis=-1)
            else:
                it_y = y[
                    idx, :, extra_vars['npoints']//2, extra_vars['size']//2,
                    extra_vars['size']//2]
            ax.plot(
                extra_vars['stimsizes'],
                it_y,
                color=extra_vars['curvecols'][idx], linewidth=5., alpha=.50)
            ax.plot(
                extra_vars['stimsizes'],
                it_y,
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
        ax.set_xlabel('Stimulus size', fontweight='bold', fontsize=15.)
        ax.set_ylabel('Cell response', fontweight='bold', fontsize=15.)
        plt.title('%s with lesion %s' % (extra_vars['figure_name'], lesion))
        plt.savefig(
            os.path.join(
                parameters._FIGURES,
                '%s_%s.pdf') % (lesion, extra_vars['figure_name']))
    elif extra_vars['figure_name'] == 'bw':
        v_contrasts = [0.0, .06, .12, .25, .50]
        h_contrasts = [0.0, .06, .12, .25, .50]
        nv, nh = len(v_contrasts), len(h_contrasts)

        def get_grid_axes():
            gs = gridspec.GridSpec(8, 5)
            ax_plots = sp.array(
                [[plt.subplot(gs[idx, jdx])
                    for jdx in range(5)] for idx in range(5)])
            gs_sub = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=gs[-2:, :])
            ax_maps = sp.array(
                [plt.subplot(gs_sub[jdx]) for jdx in range(3)])
            return ax_plots, ax_maps

        ax, _ = get_grid_axes()
        for idx in range(nv):
            for jdx in range(nh):
                ax[idx, jdx].plot(
                    y[idx, jdx, :],
                    color='#242424', linewidth=5., alpha=.50)

                ax[idx, jdx].set_ylim((0.0, y.max()))
                ax[idx, jdx].set_yticks([])

        for idx, v in enumerate(v_contrasts):
            ax[idx, 0].set_ylabel(
                '%i%%' % int(100*v),
                fontweight='bold', fontsize=15.)
            ax[idx, -1].yaxis.set_label_position('right')

        for jdx, h in enumerate(h_contrasts):
            ax[0, jdx].set_xlabel(
                '%i%%' % int(100*h),
                fontweight='bold', fontsize=15.)
            ax[0, jdx].xaxis.set_label_position('top')

        ax[nv//2, -1].set_ylabel(
            'Population responses (normalized)',
            fontweight='bold', fontsize=15.)
        ax[-1, nh//2].set_xlabel(
            'Preferred orientation (degrees)',
            fontweight='bold', fontsize=15.)
        plt.show()
        plt.close('all')
    elif extra_vars['figure_name'] == 'tbp':
        fig, ax = plt.subplots(2, 6)
        for xi in range(2):
            for yi in range(6):
                ax[xi, yi].plot(y[xi, yi, :])
        plt.show()
        plt.close(fig)
    elif extra_vars['figure_name'] == 'tbtcso':
        fig, ax_sim = plt.subplots(1, 6)
        for it_y, it_ax in zip(y, ax_sim):
            it_ax.set_xticklabels([])
            it_ax.plot(it_y)
        plt.show()
        plt.close(fig)
    elif extra_vars['figure_name'] == 'f3a':
        fig = plt.figure()
        plt.plot(y)
        plt.show()
        plt.close(fig)
    else:
        print(
            'Plotting functions not in place for your figure: %s'
            % extra_vars['figure_name'])
