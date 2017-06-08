from __future__ import absolute_import
import sys,os,argparse
sys.path.append('/home/drew/Documents/')
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import scipy as sp
from hmax.tools.utils import iround
from ops.parameter_defaults import PaperDefaults
from ops.daemon_db_hp_optim import optimize_model, adjust_parameters
from ops.fig_5_utils import *
from ops.model_utils import sfit

def run(hps=None):
    defaults = PaperDefaults()

    #David's globals
    size = 101
    mpp = 0.35 # 0.76 # 1.11
    scale = 0.12 # 0.23 # 0.22
    csv_file_x = os.path.join(defaults._DATADIR, 'WL1987_corrected_X.csv')
    csv_file_y = os.path.join(defaults._DATADIR, 'WL1987_corrected_Y.csv')

    # experiment parameters
    dd = (-150., 150.) # in seconds of arc
    sec2u = lambda s: (s - dd[0]) / (dd[1] - dd[0])
    u2sec = lambda u: u * (dd[1] - dd[0]) + dd[0]
    min2pix = lambda m: iround(m/float(mpp))
    npoints = 50
    ndists = 13
    dists = sp.linspace(0.0, 12., ndists)
    lh, lw = .25, 4.
    ph, pw = 1., 1.
    center_disp = 0.0
    flanker_disp = -33.3
    mp0 = size//2

    # Need to scale up the ecrfs 
    defaults._DEFAULT_PARAMETERS['srf'] = defaults._DEFAULT_PARAMETERS['srf'] * 2 - 1
    defaults._DEFAULT_PARAMETERS['ssn'] = defaults._DEFAULT_PARAMETERS['ssn'] * 2 - 1
    defaults._DEFAULT_PARAMETERS['ssf'] = defaults._DEFAULT_PARAMETERS['ssf'] * 2 - 1

    # simulate populations
    im = get_wl87_stim(size=size, dists=min2pix(dists),
        cval=sec2u(center_disp), sval=sec2u(flanker_disp),
        ch=min2pix(lh), cw=min2pix(lw), sh=min2pix(ph), sw=min2pix(pw))

    # Get ground truth data
    paper_data_x = sp.genfromtxt(csv_file_x, delimiter=',')
    paper_data_y = sp.genfromtxt(csv_file_y, delimiter=',') * -1
    paper_fit_y = sfit(sp.linspace(dists.min(), dists.max(), 100), paper_data_x,
        sp.nanmean(paper_data_y, axis=0), k=2, t=[5.])

    extra_vars = {}
    extra_vars['scale'] = scale
    extra_vars['kind'] = 'gaussian'
    extra_vars['decoder'] = 'circular_vote'
    extra_vars['npoints'] = npoints
    extra_vars['cval'] = sec2u(center_disp)
    extra_vars['sval'] = sec2u(flanker_disp)
    extra_vars['figure_name'] = 'f5'
    extra_vars['u2sec'] = u2sec
    extra_vars['min2pix'] = min2pix
    extra_vars['dists'] = dists
    extra_vars['flanker_disp'] = flanker_disp
    extra_vars['mp0'] = mp0
    extra_vars['lh'] = lh
    extra_vars['pw'] = pw
    extra_vars['size'] = size
    extra_vars['gt_x'] = paper_data_x
    extra_vars['return_var'] = 'I'

    defaults = adjust_parameters(defaults,hps)
    optimize_model(im,paper_fit_y,extra_vars,defaults)

if __name__ == '__main__':
    run()
