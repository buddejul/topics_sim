# Functions for plotting
import matplotlib.pyplot as plt
from funcs import *
from iv_part_id import *

# Replicate figure 1 from the paper
def plot_fig(
        identif, # the set of identified estimands
        basis, # the basis function to use for approximation
        target = None, # the target estimand
        u_lo_target = None, # if target = LATE: lower u
        u_hi_target = None, # if target = LATE: upper u
        u_lo_id = None, # if identif = late: lower u
        u_hi_id = None, # if identif = late: upper u
        u_part = None, # if basis = cs: partition of u in [0,1]
        supp_z = None, # if identif = iv_slope: support of the instrument
        prop_z = None, # if identif = iv_slope: propensity given the instrument
        f_z = None, # if identif = iv_slope: probability mass function of the instrument
        plot_weights_id = True,
        plot_weights_target = True,
        plot_dgp = True,
        m0_dgp = None,
        m1_dgp = None,
        print_bounds = True,
        quiet = True
        ):
    
    u = np.linspace(0, 1, 1000)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Get partial indentification results
    if target != None:
        results = iv_part_id(
            target = target,
            identif = identif,
            basis = basis,
            m0_dgp=m0_dgp,
            m1_dgp=m1_dgp,
            u_lo_target = u_lo_target,
            u_hi_target = u_hi_target,
            u_part = u_part,
            supp_z = supp_z,
            f_z=f_z,
            prop_z=prop_z,
            quiet = quiet
        )

        beta_lo, beta_hi = results[0][0], results[1][0]


    if plot_weights_id == False and plot_weights_target == False and plot_dgp == False:
        raise ValueError("At least one of plot_weights_id, plot_weights_target, and plot_weights_dgp must be True")

    # Compute weights of identified estimands
    if plot_weights_id == True:
        weights_id = np.zeros((len(u), 2))
        for i in range(len(u)):
            if identif == "late":
                weights_id[i, 0] = s_late(0, u[i], u_lo_id, u_hi_id)
                weights_id[i, 1] = s_late(1, u[i], u_lo_id, u_hi_id)

            if identif == "iv_slope":

                ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)

                def weight_iv_slope0(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] < u: return s_iv_slope(z, ez, cov_dz)
                    else : return 0

                def weight_iv_slope1(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] > u: return s_iv_slope(z, ez, cov_dz)
                    else : return 0

                # Evaluate weight_iv_slope1 over u grid for each z in supp_z then add them up
                weights_id[i, 0] = np.sum([weight_iv_slope0(u[i], z) * f_z[np.where(supp_z == z)[0][0]] for z in supp_z])
                weights_id[i, 1] = np.sum([weight_iv_slope1(u[i], z) * f_z[np.where(supp_z == z)[0][0]] for z in supp_z])
                        
        weights_id[np.isclose(weights_id, 0)] = np.nan
    
    if plot_weights_target == True:
        weights_target = np.zeros((len(u), 2))
        for i in range(len(u)):
            if target == "late":
                weights_target[i, 0] = s_late(0, u[i], u_lo_target, u_hi_target)
                weights_target[i, 1] = s_late(1, u[i], u_lo_target, u_hi_target)

            if target == "iv_slope":

                ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)

                def weight_iv_slope0(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] < u: return s_iv_slope(z, ez, cov_dz)
                    else : return 0

                def weight_iv_slope1(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] > u: return s_iv_slope(z, ez, cov_dz)
                    else : return 0

                # Evaluate weight_iv_slope1 over u grid for each z in supp_z then add them up
                weights_target[i, 0] = np.sum([weight_iv_slope0(u[i], z) * f_z[np.where(supp_z == z)[0][0]] for z in supp_z])
                weights_target[i, 1] = np.sum([weight_iv_slope1(u[i], z) * f_z[np.where(supp_z == z)[0][0]] for z in supp_z])
                        
        weights_target[np.isclose(weights_target, 0)] = np.nan

    # Plot for d=0
    axs[0].set_title("$d=0$")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(-4, 4)

    if plot_weights_id == True:
        axs[0].plot(u, weights_id[:, 0], label = 'Weights' + ' (' + identif + ')')

    if plot_weights_target == True:
        axs[0].plot(u, weights_target[:, 0], linestyle="--", label = 'Weights' + ' (' + target + ')')

    if plot_dgp == True:
        axs0_twin = axs[0].twinx()
        axs0_twin.plot(u, m0_dgp(u), color="black", linestyle="--", label = 'DGP')
        axs0_twin.set_ylim(0, 1)
        axs0_twin.set_xlim(0, 1)
        axs0_twin.spines['right'].set_visible(False)
        axs0_twin.spines['top'].set_visible(False)

    # Plot for d=1
    axs[1].set_title("$d=1$")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(-4, 4)

    if plot_weights_id == True:
        axs[1].plot(u, weights_id[:, 1])

    if plot_weights_target == True:
        axs[1].plot(u, weights_target[:, 1], linestyle="--")

    if plot_dgp == True:
        axs1_twin = axs[1].twinx()
        axs1_twin.plot(u, m1_dgp(u), color="black", linestyle="--")
        axs1_twin.set_ylim(0, 1)
        axs1_twin.set_xlim(0, 1)
        axs1_twin.spines['right'].set_visible(False)
        axs1_twin.spines['top'].set_visible(False)

    # Add legend outside the plot region
    fig.legend(loc='lower center', ncol=3)
    fig.subplots_adjust(bottom=0.2)

    # Add upper and lower bound from results as text above plot
    if target != None and print_bounds == True:
        fig.text(0.5, 1, 'Partial Identification Bounds: [' + str(round(beta_lo, 3)) + ',' + str(round(beta_hi, 3)) + ']', ha='center', va='center')

    plt.show()

