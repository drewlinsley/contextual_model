import os
import numpy as np
from ops.db_utils import get_all_lesion_data
from ops.parameter_defaults import PaperDefaults
from ops.plot_hp_optims import plot_chart, plot_distributions
from scipy import stats
import operator
"""
1. x Extract all data from the database
2. x Find the hyperparameters for each lesion with max fit t-scored across
problems. Send to matrix X (figures/lesions)
3. For each row of X, find the max fit, bootstrap 10x times.
4. Color all columns in a given row according to whether or not they fall in
the CI of the cell IDed in #3
"""


def perm_test(ta, tb, iterations=10000):
    T = np.asarray(ta) - np.asarray(tb)
    T_len = len(T)
    Tm = T.mean()
    diffs = np.zeros((iterations))
    for idx in range(iterations):
        signs = np.sign(np.random.rand(T_len) - 0.5)
        diffs[idx] = np.mean(T * signs)
    p_val = (np.sum(Tm < diffs) + 1).astype(float) / np.asarray(iterations + 1).astype(float)
    return p_val


def find_best_fit(lesion_data, defaults):
    zs = np.zeros((len(lesion_data)))
    mus = np.zeros((len(lesion_data)))
    for idx, row in enumerate(lesion_data):
        figure_fits = [row[k] for k in defaults.db_problem_columns]
        # if None in figure_fits:
        #   raise TypeError
        # summarize with a z score
        zs[idx] = np.nanmean(figure_fits) / np.nanstd(figure_fits)
        mus[idx] = np.nanmax(figure_fits)
    if len(zs) > 0:
        return lesion_data[np.argmax(zs)], lesion_data, zs, mus
    else:
        return None, None, None, None


defaults = PaperDefaults()
max_row = {}
lesion_dict = {}
score_dict = {}
mus = {}
for lesion in defaults.lesions:
    lesion_data = get_all_lesion_data(lesion)
    it_max, lesion_dict[lesion], score_dict[lesion], mus[lesion] = find_best_fit(
        lesion_data, defaults)
    if it_max is not None:
        it_max = dict(it_max)
    max_row[lesion] = it_max
    # bootstrap here
max_zs = {k: v.max() for k, v in score_dict.iteritems()}
# Reorder the dimensions
t_stats = {}
p_values = {}
if len(max_zs.keys()) > 1:
    new_order = sorted(max_zs.items(), key=operator.itemgetter(1))[::-1]
    new_order = [tup[0] for tup in new_order]
    new_order.pop(new_order.index('None'))
    new_order = ['None'] + new_order
    defaults.lesions = new_order
    print defaults.lesions
    target_key = 'None'

    # Run stats -- None versus the rest
    Tvs = max_row[target_key]
    T = [np.arctanh(Tvs[col]) for col in defaults.db_problem_columns]
    t_keys = [k for k in defaults.lesions if k is not target_key]
    for k in t_keys:
        t = [np.arctanh(
            max_row[k][col]) for col in defaults.db_problem_columns]
        tv, pv = stats.ttest_rel(T, t)
        t_stats[k] = tv
        p_values[k] = pv
        # p_values[k] = perm_test(T, t)
    print [(k, v) for k, v in p_values.iteritems()]
    print [(k, v) for k, v in t_stats.iteritems()]
else:
    target_key = max_zs.keys()[0]

np.savez(
    os.path.join(
        defaults._FIGURES, 'best_hps'),
    lesions=defaults.lesions,
    max_row=max_row,
    p_values=p_values,
    t_stats=t_stats)
# Remove Empty "lesions"
for v in max_row.values():
    if v is None:
        raise RuntimeError('Found empty row in your lesion matrix.')
# max_row = {k: v for k, v in max_row.iteritems() if v is not None}
# defaults.lesions = max_row.keys()

# If desired purge specific figures
if defaults.remove_figures is not None:
    for r in max_row.keys():
        for k in defaults.remove_figures:
            print 'Ommitting figure: %s' % k
            max_row[r].pop(k, None)
    defaults.db_problem_columns = list(
        set(defaults.db_problem_columns) - set(defaults.remove_figures))

plot_chart(max_row, defaults)  # also include bootstrapped CIs
plot_distributions(lesions=lesion_dict[target_key], defaults=defaults)
