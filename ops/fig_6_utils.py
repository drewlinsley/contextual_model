import scipy as sp

# get the probabilities for Multinoulli random variable
#######################################################
def get_p(direction, coherence, dirs, ndirs):
    """"""
    assert direction in dirs
    assert 0.0 <= coherence <= 1.0
    idx = sp.where(dirs == direction)[0][0]
    pvals = sp.zeros((ndirs,))
    pvals[:] = (1.0 - coherence) / float(ndirs)
    pvals[idx] = (1.0 + coherence*(float(ndirs) - 1.0)) / float(ndirs)
    return pvals

# random dot stimulus generator
###############################
def get_rdots(cval, ccoh, sval, scoh, density, size, w, dirs, ndirs):
    """"""
    im = sp.zeros((size, size))
    gx, gy = sp.mgrid[-size//2+1:size//2+1, -size//2+1:size//2+1]
    gr = sp.maximum(sp.absolute(gx), sp.absolute(gy))
    mask_c = (gr < w/2.0)
    mask_s = (gr < w) * (gr >= w/2.0)

    # get random dot locations first
    mask_c_dot = sp.zeros((size, size), dtype=bool)
    mask_s_dot = sp.zeros((size, size), dtype=bool)
    mask_c_dot[mask_c] = sp.random.choice([False, True],
        p=[1.0 - density, density], size=(mask_c.sum(),))
    mask_s_dot[mask_s] = sp.random.choice([False, True],
        p=[1.0 - density, density], size=(mask_s.sum(),))

    # then fill in with stimulus values (directions)
    im[mask_c_dot] = sp.random.choice(dirs,
        p=get_p(cval, ccoh, dirs, ndirs), size=(mask_c_dot.sum(),))
    im[mask_s_dot] = sp.random.choice(dirs,
        p=get_p(sval, scoh, dirs, ndirs), size=(mask_s_dot.sum(),))

    return im

# helper functions to fit logistic curve to the perceptual shift
###############################################################
def logistic_full(t, beta0, beta1, alpha0, alpha1):
    return alpha0/(1.0 + sp.exp(beta0*(t - beta1))) + alpha1

def logistic(t, beta0, beta1):
    alpha0 = value_up - value_down
    alpha1 = value_down
    return logistic_full(t, beta0, beta1, alpha0, alpha1)

def gaussian(t, alpha, mu, sigma):
    return alpha * sp.exp(-(t - mu)**2/sigma**2)

def fit_logistic(x, y, x_eval):
    try:
        beta0, beta1 = curve_fit(logistic, x, y, p0=[5.0, 0.0])[0]
    except RuntimeError:
        beta0, beta1 = [sp.nan, sp.nan]
    finally:
        y_eval = logistic(x_eval, beta0, beta1)
    return beta0, beta1, y_eval

def fit_logistic_full(x, y, x_eval):
    try:
        beta0, beta1, alpha0, alpha1 = curve_fit(logistic_full, x, y,
            p0=[5.0, 0.0, value_up-value_down, value_down])[0]
    except RuntimeError:
        beta0, beta1 = curve_fit(logistic, x, y, p0=[5.0, 0.0])[0]
        alpha0, alpha1 = value_up-value_down, value_down
    finally:
        y_eval = logistic_full(x_eval, beta0, beta1, alpha0, alpha1)
    return beta0, beta1, alpha0, alpha1, y_eval

def fit_gaussian(x, y, x_eval):
    try:
        alpha, mu, sigma = curve_fit(gaussian, x, y,
            p0=[y.max(), x[y.argmax()], 0.5])[0]
    except RuntimeError:
        alpha, mu, sigma = [sp.nan, sp.nan, sp.nan]
    finally:
        y_eval = gaussian(x_eval, alpha, mu, sigma)
    return alpha, mu, sigma, y_eval

