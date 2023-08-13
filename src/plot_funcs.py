# Functions for plotting
import matplotlib.pyplot as plt
from funcs import *
from iv_part_id import *

# Replicate figure 1 from the paper
def plot_fig(
        identif, # the set of identified estimands
        basis, # the basis function to use for approximation
        shape = None, # shape restriction
        target = None, # the target estimand
        u_lo_target = None, # if target = LATE: lower u
        u_hi_target = None, # if target = LATE: upper u
        u_lo_id = None, # if identif = late: lower u
        u_hi_id = None, # if identif = late: upper u
        u_part = None, # if basis = cs: partition of u in [0,1]
        supp_z = None, # if identif = iv_slope: support of the instrument
        prop_z = None, # if identif = iv_slope: propensity given the instrument
        f_z = None, # if identif = iv_slope: probability mass function of the instrument
        dz_cross = None, # if identif = cross: the cross moment
        plot_weights_id = True,
        plot_weights_target = True,
        plot_dgp = True,
        m0_dgp = None,
        m1_dgp = None,
        print_bounds = True,
        analyt_int = False,
        quiet = True
        ):
    
    u = np.linspace(0, 1, 1000)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    n_late = len([i for i in identif if "late" in i])
    n_iv_slope = len([i for i in identif if "iv_slope" in i])
    n_ols_slope = len([i for i in identif if "ols_slope" in i])
    n_cross = len([i for i in identif if "cross" in i])

    if n_iv_slope + n_ols_slope > 1 and n_iv_slope + n_ols_slope != len(supp_z):
        raise ValueError("Need to supply supp_z for each slope coefficient")
    
    if n_iv_slope + n_ols_slope > 1 and n_iv_slope + n_ols_slope != len(prop_z):
        raise ValueError("Need to supply prop_z for each slope coefficient")
    
    if n_iv_slope + n_ols_slope > 1 and n_iv_slope + n_ols_slope != len(f_z):
        raise ValueError("Need to supply f_z for each slope coefficient")
    
    if n_cross > 0 and dz_cross == None:
        raise ValueError("Need to supply dz_cross for cross moments")

    # if shape not in ["incr", "decr", None]:
    #     raise ValueError("shape must be incr, decr, or None")

    # Get partial indentification results
    if target != None:
        results = iv_part_id(
            target = target,
            identif = identif,
            shape = shape,
            basis = basis,
            m0_dgp=m0_dgp,
            m1_dgp=m1_dgp,
            u_lo_target = u_lo_target,
            u_hi_target = u_hi_target,
            u_part = u_part,
            supp_z = supp_z,
            f_z=f_z,
            dz_cross = dz_cross,
            prop_z=prop_z,
            analyt_int=analyt_int,
            quiet = quiet
        )

        beta_lo, beta_hi = results[0][0], results[1][0]


    if plot_weights_id == False and plot_weights_target == False and plot_dgp == False:
        raise ValueError("At least one of plot_weights_id, plot_weights_target, and plot_weights_dgp must be True")

    # Compute weights of identified estimands
    weights_id = []

    if plot_weights_id == True:
        weight = np.zeros((len(u), 2))

        if n_late == 1:
            for i,j in enumerate(u):
                weight[i, 0] = s_late(0, j, u_lo_id, u_hi_id)
                weight[i, 1] = s_late(1, j, u_lo_id, u_hi_id)

            weights_id.append(weight)

        elif n_late > 1:
            for i in range(n_late):
                weight = np.zeros((len(u), 2))
                for j,k in enumerate(u):
                    weight[j, 0] = s_late(0, k, u_lo_id[i], u_hi_id[i])
                    weight[j, 1] = s_late(1, k, u_lo_id[i], u_hi_id[i])

                weights_id.append(weight)

        if n_iv_slope > 0:
            def weight_iv_slope0(u, z, supp_z, prop_z): 
                if prop_z[np.where(supp_z == z)[0][0]] < u: return s_iv_slope(z, ez, cov_dz)
                else : return 0

            def weight_iv_slope1(u, z, supp_z, prop_z): 
                if prop_z[np.where(supp_z == z)[0][0]] > u: return s_iv_slope(z, ez, cov_dz)
                else : return 0

        if n_iv_slope == 1:
            weight = np.zeros((len(u), 2))
            for i,j in enumerate(u):
                ez, ed, edz, cov_dz = compute_moments(supp_z[0], f_z[0], prop_z[0])

                # Evaluate weight_iv_slope1 over u grid for each z in supp_z then add them up
                weight[i, 0] = np.sum([weight_iv_slope0(u[i], z, supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])
                weight[i, 1] = np.sum([weight_iv_slope1(u[i], z, supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])
                 

            weights_id.append(weight)

        elif n_iv_slope > 1:
            for i in range(n_iv_slope):
                weight = np.zeros((len(u), 2))
                for j,k in enumerate(u):
                    ez, ed, edz, cov_dz = compute_moments(supp_z[i], f_z[i], prop_z[i])

                    # Evaluate weight_iv_slope1 over u grid for each z in supp_z[i] then add them up
                    weight[j, 0] = np.sum([weight_iv_slope0(u[j], z, supp_z[i], prop_z[i]) * f_z[i][np.where(supp_z[i] == z)[0][0]] for z in supp_z[i]])
                    weight[j, 1] = np.sum([weight_iv_slope1(u[j], z, supp_z[i], prop_z[i]) * f_z[i][np.where(supp_z[i] == z)[0][0]] for z in supp_z[i]])

                weights_id.append(weight)
        
        if n_ols_slope > 0:
            def weight_ols_slope0(u, z, supp_z, prop_z): 
                if prop_z[np.where(supp_z == z)[0][0]] < u: return s_ols_slope(0, ed, var_d)
                else : return 0

            def weight_ols_slope1(u, z, supp_z, prop_z): 
                if prop_z[np.where(supp_z == z)[0][0]] > u: return s_ols_slope(1, ed, var_d)
                else : return 0

        if n_ols_slope == 1:
            weight = np.zeros((len(u), 2))
            for i,j in enumerate(u):
                ols_idx = 0 + n_iv_slope
                ez, ed, edz, cov_dz = compute_moments(supp_z[ols_idx], f_z[ols_idx], prop_z[ols_idx])
                var_d = ed * (1 - ed)
                weight[i, 0] = np.sum([weight_ols_slope0(u[i], z, supp_z[ols_idx], prop_z[ols_idx]) * f_z[ols_idx][np.where(supp_z[ols_idx] == z)[0][0]] for z in supp_z[ols_idx]])
                weight[i, 1] = np.sum([weight_ols_slope1(u[i], z, supp_z[ols_idx], prop_z[ols_idx]) * f_z[ols_idx][np.where(supp_z[ols_idx] == z)[0][0]] for z in supp_z[ols_idx]])
                 

            weights_id.append(weight)

        elif n_ols_slope > 1:
            for i in range(n_ols_slope):
                weight = np.zeros((len(u), 2))
                for j,k in enumerate(u):
                    ez, ed, edz, cov_dz = compute_moments(supp_z[i], f_z[i], prop_z[i])
                    var_d = ed * (1 - ed)

                    weight[j, 0] = np.sum([weight_ols_slope0(u[j], z, supp_z[i], prop_z[i]) * f_z[i][np.where(supp_z[i] == z)[0][0]] for z in supp_z[i]])
                    weight[j, 1] = np.sum([weight_ols_slope1(u[j], z, supp_z[i], prop_z[i]) * f_z[i][np.where(supp_z[i] == z)[0][0]] for z in supp_z[i]])

                weights_id.append(weight)

        if n_cross > 0:
            def weight_cross0(u, z, dz_cross, supp_z, prop_z): 
                if prop_z[np.where(supp_z == z)[0][0]] < u: return s_cross(0, z, dz_cross)
                else : return 0

            def weight_cross1(u, z, dz_cross, supp_z, prop_z): 
                if prop_z[np.where(supp_z == z)[0][0]] > u: return s_cross(1, z, dz_cross)
                else : return 0

        if n_cross == 1:
            weight = np.zeros((len(u), 2))
            for i,j in enumerate(u):
                weight[i, 0] = np.sum([weight_cross0(u[i], z, dz_cross[0], supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])
                weight[i, 1] = np.sum([weight_cross1(u[i], z, dz_cross[0], supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])

            weights_id.append(weight)

        elif n_cross > 1:
            for i in range(n_cross):
                weight = np.zeros((len(u), 2))
                for j,k in enumerate(u):
                    weight[j, 0] = np.sum([weight_cross0(u[j], z, dz_cross[i], supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])
                    weight[j, 1] = np.sum([weight_cross1(u[j], z, dz_cross[i], supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])

                weights_id.append(weight)

    for i in range(len(weights_id)):
        weights_id[i][np.isclose(weights_id[i], 0)] = np.nan
            
    if plot_weights_target == True:
        weights_target = np.zeros((len(u), 2))
        for i in range(len(u)):
            if target == "late":
                weights_target[i, 0] = s_late(0, u[i], u_lo_target, u_hi_target)
                weights_target[i, 1] = s_late(1, u[i], u_lo_target, u_hi_target)

            if target == "iv_slope":

                ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)

                def weight_iv_slope0(u, z, supp_z): 
                    if prop_z[np.where(supp_z == z)[0][0]] < u: return s_iv_slope(z, ez, cov_dz)
                    else : return 0

                def weight_iv_slope1(u, z, supp_z): 
                    if prop_z[np.where(supp_z == z)[0][0]] > u: return s_iv_slope(z, ez, cov_dz)
                    else : return 0

                # Evaluate weight_iv_slope1 over u grid for each z in supp_z then add them up
                weights_target[i, 0] = np.sum([weight_iv_slope0(u[i], z, supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])
                weights_target[i, 1] = np.sum([weight_iv_slope1(u[i], z, supp_z[0], prop_z[0]) * f_z[0][np.where(supp_z[0] == z)[0][0]] for z in supp_z[0]])
                        
        weights_target[np.isclose(weights_target, 0)] = np.nan

    # Plot for d=0
    axs[0].set_title("$d=0$")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(-4, 4)

    if plot_weights_id == True:
        if len(identif) == 1:
            axs[0].plot(u, weights_id[0][:, 0], label = 'Weights' + ' (' + identif[0] + ')')
        if len(identif) > 1:
            for i in range(len(weights_id)):
                axs[0].plot(u, weights_id[i][:, 0], label = 'Weights' + ' (' + identif[i] + ')')

    if plot_weights_target == True:
        axs[0].plot(u, weights_target[:, 0], linestyle="--", label = 'Weights' + ' (' + target + ', target)')

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
        if len(identif) == 1:
            axs[1].plot(u, weights_id[0][:, 1])
        if len(identif) > 1:
            for i in range(len(weights_id)):
                axs[1].plot(u, weights_id[i][:, 1])

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

