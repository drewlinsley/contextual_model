import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm


def decode_lesion(it_lesion):
    if len(it_lesion) == 0:
        dec_lesion = 'None'
    elif it_lesion == 'T':
        dec_lesion = 'far eCRF'
    elif it_lesion == 'Q':
        dec_lesion = 'cRF excitation'
    elif it_lesion == 'U':
        dec_lesion = 'cRF inhibition'
    elif it_lesion == 'P':
        dec_lesion = 'near eCRF'
    return dec_lesion


def extract_info(files):
    lesions = []
    params = []
    scores = []
    ids = []
    print('Extracting data')
    for idx, f in enumerate(files):
        it_lesion = re.split('.npz', re.split('lesion_', f)[-1])[0]
        dec_lesion = decode_lesion(it_lesion)
        lesions.append(dec_lesion)
        data = np.load(f)
        params.append(data['params'])
        scores.append(data['scores'])
        ids = np.append(ids, np.repeat(dec_lesion, len(scores[idx])))
    print('Done')
    return lesions, params, scores, ids


def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)


def hinton(
    W, xticks='', yticks='', xlabel='Lesion',
        ylabel='Figure name', maxweight=None, colors=['white', 'red']):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    fig, ax = plt.subplots()
    ax.set_xticklabels(xticks)
    ax.set_xticklabels(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    height, width = W.shape
    if not maxweight:
        maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    plt.fill(np.array([0, width, width, 0]),
             np.array([0, 0, height, height]),
             'gray')

    plt.axis('off')
    plt.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0:  # and w < 0.5: #convert this to CIs
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      'white')
            elif w > 2.8:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      colors[0])
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight)/4,
                      colors[1])
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, -w/maxweight),
                      'black')


def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(
            pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):  # incorporate the CI here
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def sebs(matrix, cis):
    sns.heatmap(matrix, annot=True)


def plot_chart(max_rows, defaults):
    dm = np.zeros((len(defaults.db_problem_columns), len(defaults.lesions)))
    cis = np.zeros((len(defaults.db_problem_columns), len(defaults.lesions)))
    for r, fig in enumerate(defaults.db_problem_columns):
        for c, lesion in enumerate(defaults.lesions):
            dm[r, c] = max_rows[lesion][fig]
            # cis[r,c] = True if dm[r,c] > max_rows['ci_low'][fig] else False
    if defaults.chart_style == 'hinton':
        hinton(
            dm, xticks=defaults.lesions, yticks=defaults.db_problem_columns,
            maxweight=1.1)  # ,cis) #colors=['red','white']
    elif defaults.chart_style == 'sebs':
        sebs(dm, cis)
    plt.savefig(os.path.join(defaults._FIGURES, 'hp_optims.pdf'))
    plt.close('all')
    return


def execute_density_plot(it_data, name, defaults, col_wrap=None):
    # Plot with seaborn
    sns.set(style="ticks")
    ax = sns.FacetGrid(it_data, col='Figure name', col_wrap=col_wrap)
    # bins = np.linspace(-1, 1, 20)
    ax = ax.map(
        sns.distplot, "Model fit", rug=False, hist=True,
        norm_hist=False, kde=False, bins=np.linspace(0, 1, 100), color="black",  # bins=bins,
        rug_kws={'alpha': 0.1, 'color': 'gray'})  # , edgecolor="r")
    # plt.yticks([0, 150, 300, 450, 600, 750, 900])
    plt.xlim([-1, 1])

    plt.savefig(
        os.path.join(
            defaults._FIGURES, 'coefficient_distribution_%s.pdf' % name))
    plt.close('all')


def execute_histogram_plot(it_data, name, defaults, ax=None):
    # Plot with seaborn
    sns.set(style="ticks")
    X = it_data['Model fit'].as_matrix().reshape(-1, 1)
    if ax is not None:
        ax.hist(X, bins=np.linspace(-1, 1, 1e4), normed=True)
    else:
        plt.hist(X, bins=np.linspace(-1, 1, 1e3), normed=True)
    plt.xlim([-1, 1])
    plt.ylim([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(
        os.path.join(
            defaults._FIGURES, 'coefficient_distribution_%s.pdf' % name))
    if ax is not None:
        return ax
    else:
        plt.close('all')


def plot_distributions(lesions, defaults):
    scores = []
    labels = []
    for fig in defaults.db_problem_columns:
        for r in lesions:
            scores = np.append(scores, r[fig])
            labels = np.append(labels, fig)
    df = pd.DataFrame(
        np.vstack((scores, pd.factorize(labels)[0].astype(int))).transpose(),
        columns=['Model fit', 'Figure name'])
    execute_density_plot(df, 'all', defaults, col_wrap=4)
    unique_labels = np.unique(df['Figure name'])
    f, axs = plt.subplots(2, 4)
    axs = axs.reshape(-1)
    for idx, (lab, ax) in enumerate(zip(unique_labels, axs)):
        it_data = df[df['Figure name'] == lab]
        execute_histogram_plot(it_data, idx, defaults, ax=ax)
        # execute_density_plot(it_data, idx, defaults)


if __name__ == '__main__':
    data_dir = 'optim_npys'
    files = sorted(glob(os.path.join(data_dir, '*.npz')))
    l, p, s, i = extract_info(files)
    max_ids = [np.argmax(x) for x in s]
    max_scores = [np.max(x) for x in s]
    best_params = [x[max_ids[idx]] for idx, x in enumerate(p)]

    # Add to a pandas df so we can plot easily
    df = pd.DataFrame(
        np.vstack((np.hstack(s), i)).transpose(),
        columns=['Model fit', 'Lesion type'])

    # Plot with seaborn
    sns.set(style="ticks")
    ax = sns.stripplot(
        x="Lesion type", y="Model fit", data=df, jitter=True, size=0.01,
        palette='Paired')
    ax.yaxis.set_label_text(
        'Pearson correlation between experimental and simulated data')
    out_file_name = '_'.join(l) + '.pdf'
    out_file_name = '_'.join(out_file_name)
    plt.savefig(out_file_name)
