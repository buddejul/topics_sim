# Running simulation in parallel (for now with different tolerances)
import multiprocessing as mp
from sim_funcs import *
import time
import numpy as np
import os
import itertools

start = time.time()

if __name__ == '__main__':
    
    bld = "C:/Users/budde/OneDrive/phd_bgse/courses/topics_metrics/topics_sim/bld/"
    # Add subfolder to bld with systemtime and create if it does not exist
    bld = bld + "sim_parallel_" + time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(bld):
        os.makedirs(bld)

    print("Number of processors: ", mp.cpu_count())
    poolsize = mp.cpu_count() - 1

    # Initialize pool of cores
    pool = mp.Pool(poolsize)

    N_vals = [25000, 50000]
    R = 500

    def get_tolerances(N):
        # return [N**(-0.5), 3*N**(-1), N**(-1), N**(-2)]
        return [N**(-1)]

    # Target parameter
    target = "late"
    u_lo_target = 0.35
    # u_hi_target = 0.9

    u_hi_targetlist = np.arange(0.4, 1, 0.025)

    basis = "cs" # the basis function to use for approximation
    supp_z = [np.array([0, 1, 2])] # if identif = iv_slope: support of the instrument
    prop_z = [np.array([0.35, 0.6, 0.7])] # if identif = iv_slope: propensity given the instrument
    f_z = [np.array([0.5, 0.4, 0.1])] # if identif = iv_slope: probability mass function of the instrument
    dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] # if identif = cross: the cross moment
    
    identif = ["cross","cross","cross","cross","cross","cross"] # the set of identified estimands

    # Write a txt file into the folder to keep track of the parameters
    with open(bld + "/parameters.txt", "w") as f:
        f.write("N_vals = " + str(N_vals) + "\n")
        f.write("R = " + str(R) + "\n")
        f.write("tolerances = [N**(-1)] \n")
        f.write("target = " + target + "\n")
        f.write("u_lo_target = " + str(u_lo_target) + "\n")
        f.write("u_hi_target_list = " + str(u_hi_targetlist) + "\n")
        f.write("identif = " + str(identif) + "\n")
        f.write("dz_cross = " + str(dz_cross) + "\n")
        f.write("basis = " + basis + "\n")
        f.write("supp_z = " + str(supp_z) + "\n")
        f.write("prop_z = " + str(prop_z) + "\n")
        f.write("f_z = " + str(f_z) + "\n")


    # Create list of N_vals, tolerances combinations
    combinations = list(itertools.product(N_vals, map(get_tolerances, N_vals)))
    combinations = [(N, tol) for N, tolerances in combinations for tol in tolerances]

    # Now for each element in combinations, create a triplet of arguments using u_hi_targetlist
    combinations = [(N, tol, u_hi_target) for N, tol in combinations for u_hi_target in u_hi_targetlist]
   
    # Print combinations into parameter.txt
    with open(bld + "/parameters.txt", "a") as f:
        f.write("combinations = " + str(combinations) + "\n")

    results = [pool.apply_async(simulation, 
                                args = (N, R, supp_z, f_z, prop_z,
                                         target, identif, basis, tol, u_lo_target,
                                                 u_hi_target, dz_cross,
                                                  False, True, bld, False, True)) for N, tol, u_hi_target in combinations]

    pool.close()
    pool.join()

end = time.time()

print("Elapsed time: " + str(end - start))