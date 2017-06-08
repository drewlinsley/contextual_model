import os
import numpy as np
from ops.db_utils import get_all_lesion_data
from ops.parameter_defaults import PaperDefaults
from ops.plot_hp_optims import plot_chart, plot_distributions
"""
1. x Extract all data from the database
2. x Find the hyperparameters for each lesion with max fit t-scored across
problems. Send to matrix X (figures/lesions)
3. For each row of X, find the max fit, bootstrap 10x times.
4. Color all columns in a given row according to whether or not they fall in
the CI of the cell IDed in #3
"""


def find_best_fit(lesion_data, defaults):
    zs = np.zeros((len(lesion_data)))
    for idx, row in enumerate(lesion_data):
        figure_fits = [row[k] for k in defaults.db_problem_columns]
        # if None in figure_fits:
        #   raise TypeError
        # summarize with a z score
        zs[idx] = np.nanmean(figure_fits) / np.nanstd(figure_fits)
    if len(zs) > 0:
        return lesion_data[np.argmax(zs)], lesion_data, zs
    else:
        return None, None, None


defaults = PaperDefaults()
max_row = {}
lesion_dict = {}
score_dict = {}
for lesion in defaults.lesions:
    lesion_data = get_all_lesion_data(lesion)
    it_max, lesion_dict[lesion], score_dict[lesion] = find_best_fit(
        lesion_data, defaults)
    if it_max is not None:
        it_max = dict(it_max)
    max_row[lesion] = it_max
    # bootstrap here

np.savez(
    os.path.join(
        defaults._FIGURES, 'best_hps'), lesions=defaults.lesions,
    max_row=max_row)

# Remove Empty "lesions"
max_row = {k: v for k, v in max_row.iteritems() if v is not None}
defaults.lesions = max_row.keys()

# If desired purge specific figures
if defaults.remove_figures is not None:
    for r in max_row.keys():
        for k in defaults.remove_figures:
            print 'Ommitting figure: %s' % k
            max_row[r].pop(k, None)
    defaults.db_problem_columns = list(
        set(defaults.db_problem_columns) - set(defaults.remove_figures))

plot_chart(max_row, defaults)  # also include bootstrapped CIs
plot_distributions(lesions=lesion_dict['None'], defaults=defaults)
