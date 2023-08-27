import numpy as np

def import_sim_params():
    
    identif = ["cross","cross","cross","cross","cross","cross"] # the set of identified estimands
    basis = "cs" # the basis function to use for approximation
    shape = None # shape restriction
    target = "late" # the target estimand
    u_lo_target = 0.35 # if target = LATE: lower u
    u_hi_target = 0.9 # if target = LATE: upper u
    supp_z = [np.array([0, 1, 2])] # if identif = iv_slope: support of the instrument
    prop_z = [np.array([0.35, 0.6, 0.7])] # if identif = iv_slope: propensity given the instrument
    f_z = [np.array([0.5, 0.4, 0.1])] # if identif = iv_slope: probability mass function of the instrument
    dz_cross = dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] # if identif = cross: the cross moment

    u_part = [np.array([0, 0.35, 0.6, 0.7, 0.9, 1])] # if basis = cs: partition of u in [0,1]

    # Pack all of this into dictionary
    sim_params = {
        "identif": identif,
        "basis": basis,
        "shape": shape,
        "target": target,
        "u_lo_target": u_lo_target,
        "u_hi_target": u_hi_target,
        "supp_z": supp_z,
        "prop_z": prop_z,
        "f_z": f_z,
        "dz_cross": dz_cross,
        "u_part": u_part
    }

    return sim_params