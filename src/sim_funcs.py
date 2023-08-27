# Simulation function
from iv_part_id_estimation import *
from iv_part_id import *
from plot_funcs import *
from funcs import *
from simulate_data import * 

# Now run a simulation
def simulation(n, r, supp_z, f_z, prop_z, target, identif, basis, tol, 
               u_lo_target=None, u_hi_target=None, dz_cross=None, quiet=True,
               save = True, savepath = None, return_gamma=False,
               u_part_fix=False):

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
            quiet = quiet,
            return_gamma = return_gamma,
            u_part_fix=u_part_fix
            )
    
        beta_hi[i], beta_lo[i] = results[0][0], results[1][0]

        if return_gamma is True:
            # Assign first df
            if i == 0:
                gamma_df = results[2]
                gamma_id_df = results[3]
                iv_df = results[4]
                p_df = results[5]

                gamma_df["i"] = i
                gamma_id_df["i"] = i

                iv_df = pd.concat(iv_df, axis=1)
                iv_df["i"] = i

                p_df = pd.DataFrame(p_df)
                p_df["i"] = i

                val_estimands = pd.DataFrame(results[6])
                val_estimands["i"] = i

            else:
                a = results[2]
                a["i"] = i
                b = results[3]
                b["i"] = i
                c = results[4]
                c = pd.concat(c, axis=1)
                c["i"] = i

                p = pd.DataFrame(results[5])
                p["i"] = i

                gamma_df = pd.concat([gamma_df, a], axis=0)
                gamma_id_df = pd.concat([gamma_id_df, b], axis=0)
                iv_df = pd.concat([iv_df, c], axis=0)
                p_df = pd.concat([p_df, p], axis=0)

                val_estimands = pd.concat([val_estimands, pd.DataFrame(results[6])], axis=0)
                val_estimands["i"] = i
                
    if save == True:
        np.save(savepath + "/beta_lo_n" + str(n) +"_r" + str(r) + "_tol" + str(tol) + "_lo" + str(u_lo_target) + "_hi" + str(u_hi_target) + ".npy", beta_lo)
        np.save(savepath + "/beta_hi_n" + str(n) +"_r" + str(r) + "_tol" + str(tol) + "_lo" + str(u_lo_target) + "_hi" + str(u_hi_target) + ".npy", beta_hi)

    if return_gamma is False: return beta_lo, beta_hi
    else: return beta_lo, beta_hi, gamma_df, gamma_id_df, iv_df, p_df, val_estimands
