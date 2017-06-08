import scipy as sp

def _plot_TrottBorn2015_tcso_data(csvfiles):

    # get digitized data points
    ###########################
    r_x = sp.zeros((6, 7))
    r_y = sp.zeros((6, 7))

    # load and wrap first point (x-wise) to last position
    for idx, csv in enumerate(csvfiles[:-1]):
        r_x[idx, :6], r_y[idx, :6] = \
            sp.genfromtxt(csv, delimiter=',').T
        r_x[idx, -1] = 90
        r_y[idx, -1] = r_y[idx, 0]

    # locus of maximal inhibition (digitized from paper)
    ####################################################
    dr = sp.array([-89., -55., -17., 0., 17., 58.])

    return r_y, dr
