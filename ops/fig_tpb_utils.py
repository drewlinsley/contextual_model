import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit

def _plot_TrottBorn2015_population_plaid_data(npoints,csvfiles):
    """"""

    # get digitized data points
    ###########################
    ps_x = sp.zeros((6, 7))
    ps_y = sp.zeros((6, 7))
    po_x = sp.zeros((6, 7))
    po_y = sp.zeros((6, 7))
    c2c1 = sp.linspace(-90, 60, 6)

    # load and wrap first point (x-wise) to last position
    for idx, (csv_ps, csv_po) in enumerate(zip(*csvfiles)):
        ps_x[idx, :-1], ps_y[idx, :-1] = \
            sp.genfromtxt(csv_ps, delimiter=',').T
        po_x[idx, :-1], po_y[idx, :-1] = \
            sp.genfromtxt(csv_po, delimiter=',').T
        ps_x[idx, -1] = 90
        po_x[idx, -1] = 90
        ps_y[idx, -1] = ps_y[idx, 0]
        po_y[idx, -1] = po_y[idx, 0]

    # decoded mean vectors, digitized from the paper
    #do = sp.array([16.67, -27.13, -12.57,  01.99,  22.50,  29.56])
    #ds = sp.array([04.67, -11.25,  00.67,  06.61,  05.96,  10.51])

    # fit: define parametric curve (Eq. 2, Trott & Born 2015)
    #########################################################

    # first estimate dispersion from special case
    # (same ori, plaid-only; see paper for details) ...
    i_nodiff = sp.where(c2c1 == 0)[0][0]
    def R_0(t_deg, w, k, kappa):
        t_rad = t_deg / 180. * sp.pi
        T0 = stats.vonmises.pdf(t_rad*2.0,
            loc=0.0, scale=1.0, kappa=kappa)
        T0 /= T0.max()
        return w * T0 - k
    _, _, kappa = curve_fit(R_0,
        xdata=po_x[i_nodiff], ydata=po_y[i_nodiff])[0]

    # ... then fit the rest
    def pf(theta_deg):
        def R_theta(t_deg, w1, w2, k):
            t_rad = t_deg / 180. * sp.pi
            theta_rad = theta_deg / 180. * sp.pi
            T0 = stats.vonmises.pdf(t_rad*2.0,
                loc=0.0, scale=1.0, kappa=kappa)
            Tt = stats.vonmises.pdf(t_rad*2.0,
                loc=theta_rad*2.0, scale=1.0, kappa=kappa)
            T0 /= T0.max()
            Tt /= Tt.max()
            return w1 * T0 + w2 * Tt - k
        return R_theta

    po_par = sp.zeros((6, 3))
    ps_par = sp.zeros((6, 3))
    for theta, x_, y_, par in zip(c2c1, po_x, po_y, po_par):
        par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
    for theta, x_, y_, par in zip(c2c1, ps_x, ps_y, ps_par):
        par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)

    po_fit = sp.zeros((6, npoints))
    ps_fit = sp.zeros((6, npoints))
    xx_rad = sp.linspace(-sp.pi, sp.pi, npoints)
    xx_deg = xx_rad / sp.pi * 90.
    for idx, (theta, par, fit) in enumerate(zip(c2c1, po_par, po_fit)):
        fit[:] = pf(theta)(xx_deg, *par)
    for idx, (theta, par, fit) in enumerate(zip(c2c1, ps_par, ps_fit)):
        fit[:] = pf(theta)(xx_deg, *par)

    return po_y, ps_y
