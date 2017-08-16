import os
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import seaborn as sns
from matplotlib import pyplot as plt
# from ops import fig_4_utils
# from ops import colormaps
# from ops import cell_figure_settings

_DATADIR = '/home/drew/Documents/dmely_hmax/models/ucircuits/contextual/data'
_DEFAULT_TILTEFFECT_CSV = {
    'ow77': os.path.join(_DATADIR, 'OW_fig4_Black.csv'),
    'ms79': os.path.join(_DATADIR, 'MS1979.csv'),
}
_, ds_ow77_paper_y = sp.genfromtxt(_DEFAULT_TILTEFFECT_CSV['ow77'], delimiter=',').T

data = np.load('f3_plot_data.npy')

ttime = np.repeat(np.arange(data.shape[1])[None, :], data.shape[0], axis=0)
ttime = ttime.reshape(-1, 1)

group = np.repeat(np.arange(len(data))[:, None], data.shape[1], axis=1)
group = group.reshape(-1, 1)

long_y = data.reshape(-1, 1)
sns.set_style("white")
plt.figure(figsize=(10, 10))
lines = plt.plot(data.transpose())
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plt.setp(lines, alpha=0.1, linewidth=0.5, color='black')
plt.savefig('multi_sub_f3a.pdf')

sns.set_style("white")
plt.figure(figsize=(10, 10))

l2 = [interp1d(np.linspace(0, data.shape[1], data.shape[1], endpoint=True), d, kind='cubic') for d in data]
ilines = [f(np.linspace(0, data.shape[1], 1001, endpoint=True)) for f in l2]

lines = plt.plot(np.asarray(ilines).transpose())
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plt.setp(lines, alpha=0.1, linewidth=0.5, color='black')

ds_ow77_paper_y /= 100.
gt_data = interp1d(np.linspace(0, len(ds_ow77_paper_y), len(ds_ow77_paper_y), endpoint=True), ds_ow77_paper_y, kind='cubic')
interp_gt = gt_data(np.linspace(0, len(ds_ow77_paper_y), 1001, endpoint=True))
line_gt = plt.plot(np.asarray(interp_gt))
plt.setp(line_gt, linewidth=0.5, color='#429bf4')
plt.savefig('multi_sub_f3a_interp.pdf')



# df = pd.DataFrame(np.hstack((ttime, group, long_y)), columns=['time', 'group', 'y'])


# # ax = sns.pointplot(x='time', y='y', hue='group', data=df, markers='None', color='black', ci=None, alpha=1e-10, size=0.01)

# f, ax = plt.subplot(1, 1)
# for idx in np.unique(group):
#     plt.plot(long_y[group==idx], time[group==idx])


# ax.legend_.remove()
# plt.show()
