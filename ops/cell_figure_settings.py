#------------------------------------------------------------------------------#
""" Cell default figure guidelines """

# uncomment if we need to render figures at hi-res
PUBLISH = True
if PUBLISH:
    import matplotlib
    matplotlib.use('PDF')

# graphics packages and settings
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable as maxl
import matplotlib.gridspec as gridspec
import seaborn as sns
import colormaps as custom_colormaps
cm.register_cmap(name='viridis', cmap=custom_colormaps.viridis)
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['axes.grid'] = False
default_cmap = 'viridis'
plt.style.use('ggplot')

CELL_LINEWIDTH = 5.
CELL_LINECOLOR = 'k'
CELL_LINEALPHA = 0.50
CELL_MARKERSIZE = 10.
CELL_MARKERCOLOR = 'k'
CELL_MARKERALPHA = 0.75
CELL_ABSCISSA_ALPHA = 0.75
CELL_ABSCISSA_C = 'k'
CELL_ABSCISSA_LW = 1.0

# adjust model plot ylims to make their abscissa axes align with papers'
def retick(ax, n, k, zeroonly=False):
    """ adjusts ax so that there are n ticks, and the k-th one is 0.0 """
    yticks = sp.linspace(
        min(ax.get_ylim()), max(ax.get_ylim()), n)
    ax.set_yticks(yticks)
    ax.set_ylim(sp.array([y for y in ax.get_ylim()]) - yticks[k])
    delta = abs(ax.get_ylim()[0])/float(k)
    yticks = [ax.get_ylim()[0] + i*delta for i in range(n)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['%.1f' % k for k in yticks])
    if zeroonly:
        yticklabels = sp.array(['' for i in range(n)])
        yticklabels[k] = '0.0'
        ax.set_yticklabels(yticklabels)

def mm2inch(*args):
    inch = 25.4
    if isinstance(args[0], tuple):
        return tuple(i/inch for i in args[0])[0]
    else:
        return tuple(i/inch for i in args)

if PUBLISH:
    CELL_DEFAULT_TICK_LABEL_SIZE = 10.
    CELL_DEFAULT_AXES_LABEL_SIZE = 10.
    CELL_DEFAULT_AXES_TITLE_SIZE = 10.
    CELL_DEFAULT_PLOT_TITLE_SIZE = 10.
    CELL_DEFAULT_PLOT_TITLE_WEIGHT = 'medium'
    CELL_DEFAULT_FIGSIZE = (mm2inch(174., 87.)) # 87 mm x 174 mm
    CELL_DEFAULT_DPI = 1200.
    CELL_DEFAULT_SAVEFIG_PAD_INCHES = 0.35

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['xtick.labelsize'] = CELL_DEFAULT_TICK_LABEL_SIZE
    plt.rcParams['ytick.labelsize'] = CELL_DEFAULT_TICK_LABEL_SIZE
    plt.rcParams['axes.labelsize'] = CELL_DEFAULT_AXES_LABEL_SIZE
    plt.rcParams['axes.titlesize'] = CELL_DEFAULT_AXES_TITLE_SIZE
    plt.rcParams['figure.figsize'] = CELL_DEFAULT_FIGSIZE
    plt.rcParams['figure.dpi'] = CELL_DEFAULT_DPI
    plt.rcParams['savefig.dpi'] = CELL_DEFAULT_DPI
    plt.rcParams['savefig.pad_inches'] = CELL_DEFAULT_SAVEFIG_PAD_INCHES

    def CELLFIG(
        fig=None,
        out=None,
        size=CELL_DEFAULT_FIGSIZE,
        dpi=CELL_DEFAULT_DPI):
        """ Save with preferences """

        if fig is None:
            fig = plt.gcf()

        fig.set_size_inches(*size)
        fig.set_dpi(dpi)
        # fig.set_tight_layout(True)
        # plt.tight_layout(pad=.40)
        plt.draw()

        if out is not None:
            fig.savefig(out, dpi=dpi, format='pdf', bbox_inches='tight',
                pad_inches=CELL_DEFAULT_SAVEFIG_PAD_INCHES)

    CELL_LINEWIDTH = 1.5 # 1.0 pt
    CELL_LINECOLOR = '#222222'
    CELL_LINEALPHA = 0.75
    CELL_MARKERSIZE = 5. # 10 pt
    CELL_MARKERCOLOR = '#3D3D3D'
    CELL_MARKERALPHA = 1.0
    CELL_ABSCISSA_ALPHA = 0.75
    CELL_ABSCISSA_C = 'k'
    CELL_ABSCISSA_LW = 1.0

# Miscellaneous
PALETTE_4COLOR = [sns.color_palette("Set2", 10)[i] for i in [1, 2, 4, 5]]
PALETTE_4BLUES = sns.color_palette("Blues", 10)[-7::2]
PALETTE_4REDS = sns.color_palette("Reds", 20)[-7::2]
DATADIR = '/home/drew/Documents/dmely-hmax/models/ucircuits/contextual/data'
WORKDIR = '/home/drew/Documents/dmely-hmax/models/ucircuits/contextual/working'

