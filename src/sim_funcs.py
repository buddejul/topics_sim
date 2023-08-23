# Simulation function
from iv_part_id_estimation import *
from iv_part_id import *
from plot_funcs import *
from funcs import *
from simulate_data import * 

# Now run a simulation
def simulation(n, r, supp_z, f_z, prop_z, target, identif, basis, tol, 
               u_lo_target=None, u_hi_target=None, dz_cross=None, quiet=True,
               save = True, savepath = None):

    beta_lo = np.zeros(r)
    beta_hi = np.zeros(r)

    if save == True and savepath == None:
        raise ValueError("Please specify a savepath.")  

    for i in range(r):
        # Print number of current run
        print("\n")
        print("\n")
        print("Run: " + str(i + 1))
        print("\n")
        print("\n")

        # Simulate data
        data = simulate_data(n, supp_z[0], f_z[0], prop_z[0])

        y = data[:, 0]
        d = data[:, 1] 
        z = data[:, 2]

        # Estimate bounds
        results = iv_part_id_estimation(
            y,
            z,
            d,
            target, 
            identif,
            basis, 
            u_lo_target = u_lo_target, 
            u_hi_target = u_hi_target,
            dz_cross = dz_cross,
            tol = tol,
            quiet = quiet
            )
        
        beta_hi[i], beta_lo[i] = results[1][0], results[2][0]

    if save == True:
        np.save(savepath + "/beta_lo_n" + str(n) +"_r" + str(r) + "_tol" + str(tol) + "_lo" + str(u_lo_target) + "_hi" + str(u_hi_target) + ".npy", beta_lo)
        np.save(savepath + "/beta_hi_n" + str(n) +"_r" + str(r) + "_tol" + str(tol) + "_lo" + str(u_lo_target) + "_hi" + str(u_hi_target) + ".npy", beta_hi)

    return beta_lo, beta_hi
