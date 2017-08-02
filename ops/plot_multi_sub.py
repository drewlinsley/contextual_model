import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# from ops import fig_4_utils
# from ops import colormaps
# from ops import cell_figure_settings


data = np.load('f3_plot_data.npy')

ttime = np.repeat(np.arange(data.shape[1])[None, :], data.shape[0], axis=0)
ttime = ttime.reshape(-1, 1)

group = np.repeat(np.arange(len(data))[:, None], data.shape[1], axis=1)
group = group.reshape(-1, 1)

long_y = data.reshape(-1, 1)

lines = plt.plot(data.transpose())
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

plt.setp(lines, alpha=0.1, linewidth=0.5, color='black')
plt.savefig('multi_sub_f3a.pdf')

# df = pd.DataFrame(np.hstack((ttime, group, long_y)), columns=['time', 'group', 'y'])


# # ax = sns.pointplot(x='time', y='y', hue='group', data=df, markers='None', color='black', ci=None, alpha=1e-10, size=0.01)

# f, ax = plt.subplot(1, 1)
# for idx in np.unique(group):
#     plt.plot(long_y[group==idx], time[group==idx])


# ax.legend_.remove()
# plt.show()