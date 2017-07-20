from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import scipy as sp
from ops import stimuli as stim
from ops.parameter_defaults import PaperDefaults
from ops.dumb_daemon_db_hp_optim import optimize_model, sampler
from ops import model_utils
from ops.fig_tpb_utils import _plot_TrottBorn2015_population_plaid_data as get_gt


def run():
    defaults = PaperDefaults()

    #David's globals
    size=51
    csize=9
    npoints=32
    scale=2.0
    cval=0.5
    csvfiles = [[ defaults._DATADIR + \
        '/TB2015_%i_%s.csv' % (i, s) \
        for i in range(-90, 90, 30)] for s in ('PS', 'PO')
    ]

    # experiment parameters
    scale = sampler(scale)
    ppop = {
        'kind': 'circular',
        'npoints': npoints,
        'scale': scale,
        'fdomain': (0, 1),
    }

    vals_ang = sp.array([-90., -60., -30., 0., 30., 60.])
    vals = (vals_ang + 90.)/180.
    imc1 = stim.get_center_surround(
        size=size, csize=csize, cval=cval, sval=sp.nan)
    x1 = model_utils.get_population(imc1, **ppop)
    x = sp.zeros((2, len(vals), npoints, size, size))
    
    for vdx, v in enumerate(vals):
        imc2 = stim.get_center_surround(
            size=size, csize=csize, cval=v, sval=sp.nan)
        ims = stim.get_center_surround(
            size=size, csize=csize, cval=sp.nan, sval=v)
        x2 = model_utils.get_population(imc2, **ppop)
        xs = model_utils.get_population(ims, **ppop)
        x[0, vdx] = (x1 + x2)/2.
        x[1, vdx] = (x1 + x2)/2. + xs
    x.shape = (2 * len(vals), npoints, size, size)

    # trott and born 2015 data
    gt = get_gt(npoints,csvfiles)

    extra_vars = {}
    extra_vars['scale'] = scale
    extra_vars['npoints'] = npoints
    extra_vars['cval'] = cval
    extra_vars['size'] = size
    extra_vars['csize'] = csize
    extra_vars['vals'] = vals
    extra_vars['figure_name'] = 'tbp'
    extra_vars['return_var'] = 'O'

    optimize_model(x,gt,extra_vars,defaults)

if __name__ == '__main__':
    run()
