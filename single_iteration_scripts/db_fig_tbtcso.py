from __future__ import absolute_import
import sys,os
sys.path.append('/home/drew/Documents/')
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import scipy as sp
from hmax.models.ucircuits.contextual import stimuli as stim
from ops.parameter_defaults import PaperDefaults
from ops.single_hp_optim_yhist import optimize_model
from ops import model_utils
from ops.fig_tpbtcso_utils import _plot_TrottBorn2015_tcso_data as get_gt

def run():
    defaults = PaperDefaults()

    #David's globals
    size=51
    csize=5
    npoints=64
    scale=2.0
    neuron_theta = 0.50
    cval=0.5
    csvfiles = [ defaults._DATADIR + \
        '/TB2015_Fig1B_%s.csv' % (s,) \
    for s in range(-90, 90, 30) + ['CO']]


    # experiment parameters
    cvals = (sp.arange(-90, 90, 30) + 90.) / 180.
    svals = sp.linspace(0.0, 1.0, 6).tolist() + [sp.nan]

    neuron_thetas = sp.linspace(0.0, 1.0, npoints)
    neuron_idx = sp.argmin(sp.absolute(neuron_thetas - neuron_theta))

    stims = [stim.get_center_surround(size=size,
        csize=csize, cval=cv, sval=sv)
        for cv in cvals for sv in svals]

    x = sp.array([model_utils.get_population(im,
            npoints=npoints,
            kind='circular',
            scale=scale,
            fdomain=(0, 1))
        for im in stims])

    # [Array shapes]
    # trott and born 2015 data
    gt = get_gt(csvfiles)

    extra_vars = {}
    extra_vars['scale'] = scale
    extra_vars['npoints'] = npoints
    extra_vars['cval'] = cval
    extra_vars['cvals'] = cvals
    extra_vars['svals'] = svals
    extra_vars['size'] = size
    extra_vars['csize'] = csize
    extra_vars['neuron_idx'] = neuron_idx
    extra_vars['figure_name'] = 'tbtcso'
    extra_vars['return_var'] = 'O'
    extra_vars['hp_file'] = os.path.join(defaults._FIGURES, 'best_hps.npz')

    optimize_model(x,gt,extra_vars,defaults)

if __name__ == '__main__':
    run()