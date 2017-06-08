from scipy import stats

def R_0(t_deg, w, k, kappa):
    t_rad = t_deg / 180. * sp.pi
    T0 = stats.vonmises.pdf(t_rad*2.0,
        loc=0.0, scale=1.0, kappa=kappa)
    T0 /= T0.max()
    return w * T0 - k

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

