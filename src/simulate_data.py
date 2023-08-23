# Functions to simulate dataimport numpy as np
from funcs import *

def simulate_data(N, supp_z, f_z, prop_z):

    # Draw uniform choice error (unobserved part)
    u = np.random.uniform(size=N)

    # Draw instrument: independent of u
    z = np.random.choice(supp_z, size=N, p=f_z)   

    # Determine treatment
    p = prop_z[z] # note this only works because z in 0,1,2 FIXME
    d = p >= u

    # Determine y based on m0_dgp and m1_dgp
    # Unobserved potential outcomes
    y0 = m0_dgp(u)
    y1 = m1_dgp(u)

    # Observed outcomes
    y = np.zeros(N)
    y[d==1] = y1[d==1]
    y[d==0] = y0[d==0]

    # Combine into one data set
    data = np.column_stack((y, d, z, u, y0, y1))

    return data