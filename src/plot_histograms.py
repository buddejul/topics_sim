from simulation_sharp_bounds import read_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from iv_part_id import iv_part_id
from funcs import m0_dgp, m1_dgp

def plot_histograms(filelo, filehi, dir_result, dir_save, separate=True):
    
    # Define parameters
    target = "late"
    identif = ["cross", "cross", "cross", "cross", "cross", "cross"]
    basis = "cs"
    u_lo_target = 0.35
    u_hi_target = 0.9

    f_z = [np.array([0.5, 0.4, 0.1])]
    supp_z = [np.array([0, 1, 2])]
    prop_z = [np.array([0.35, 0.6, 0.7])]

    dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    u_part = [np.array([0, 0.35, 0.6, 0.7, 0.9, 1])]

    # Step 1: Get different bounds by target parameter
    truth = iv_part_id(
        target, 
        identif,
        basis, 
        m0_dgp,
        m1_dgp,
        shape=None,
        k0=None, 
        k1=None, 
        u_part = u_part[0], 
        u_lo_target = u_lo_target, 
        u_hi_target = u_hi_target,
        supp_z = supp_z,
        prop_z = prop_z,
        f_z = f_z,
        dz_cross = dz_cross,
        analyt_int = True,
        quiet = False
    )

    # Now extract the bounds from the out
    beta_lo_true = truth[0][0]
    beta_hi_true = truth[1][0]
    beta_lo_true, beta_hi_true

    # Read in all files in the folder results
    df_est_lo = read_results(dir_result, filelo)
    df_est_hi = read_results(dir_result, filehi)

    # Append df_est_lo and df_est_hi
    df_est = df_est_lo.append(df_est_hi)
    df_est.head()

    if separate is False:
        # Create a single plot overlaying two histograms, one for beta_lo, one for beta_hi
        plt.hist(df_est[df_est['bound'] == 'beta_lo']['beta'], bins='auto', density=True, alpha=0.5, label='Estimated Lower Bound')
        plt.hist(df_est[df_est['bound'] == 'beta_hi']['beta'], bins='auto', density=True, alpha=0.5, label='Estimated Upper Bound')

        # Add the means of the two distributions
        plt.axvline(x=beta_lo_true, alpha = 0.5, color='blue', label='')
        plt.axvline(x=beta_hi_true, alpha = 0.5, color='red', label='')

        # Add means of estimated distributions
        plt.axvline(x=df_est[df_est['bound'] == 'beta_lo']['beta'].mean(), alpha = 0.5, color='blue', linestyle='dashed', label='')
        plt.axvline(x=df_est[df_est['bound'] == 'beta_hi']['beta'].mean(), alpha = 0.5, color='red', linestyle='dashed', label='')


        # Overlay a normal distributoin with the same mean and variance as the estimated distribution
        # for the lower bound
        mu_lo = df_est[df_est['bound'] == 'beta_lo']['beta'].mean()
        sigma_lo = df_est[df_est['bound'] == 'beta_lo']['beta'].std()
        x = np.linspace(mu_lo - 3*sigma_lo, mu_lo + 3*sigma_lo, 100)
        plt.plot(x, norm.pdf(x, mu_lo, sigma_lo), alpha=0.5, color='blue', linestyle='dashed', label='')

        # Overlay a normal distributoin with the same mean and variance as the estimated distribution
        # for the upper bound
        mu_hi = df_est[df_est['bound'] == 'beta_hi']['beta'].mean()
        sigma_hi = df_est[df_est['bound'] == 'beta_hi']['beta'].std()
        x = np.linspace(mu_hi - 3*sigma_hi, mu_hi + 3*sigma_hi, 100)
        plt.plot(x, norm.pdf(x, mu_hi, sigma_hi), alpha=0.5, color='red', linestyle='dashed', label='')


        # Add a legend but only for the first two entries
        plt.legend()

        # Move legend outside of plot region
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        # Get n and tol from dataframe
        n = df_est['n'].iloc[0]
        tol = df_est['tol'].iloc[0]
        r = df_est['r'].iloc[0]


        # Add N and tol as a note in the plot
        plt.text(0.95, 1, 'N = ' + str(n) + ',' + 'tol = ' + str(tol) + ',' + 'R = ' + str(r),
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=plt.gca().transAxes)


        # Save graph and add n, r, kappa_n to the filename
        plt.savefig(dir_save + "/histo_" + str(n) + "_" + str(r) + "_" + str(tol) + ".png", dpi=300)

    if separate is True:
        # Make separate plots for beta_lo and beta_hi
        # Create a single plot overlaying two histograms, one for beta_lo, one for beta_hi
        plt.hist(df_est[df_est['bound'] == 'beta_lo']['beta'], color = 'blue', bins='auto', density=True, alpha=0.5, label='Estimated Lower Bound')
        # plt.axvline(x=beta_lo_true, alpha = 0.5, color='blue', label='')
        plt.axvline(x=df_est[df_est['bound'] == 'beta_lo']['beta'].mean(), alpha = 0.5, color='blue', linestyle='dashed', label='')

        mu_lo = df_est[df_est['bound'] == 'beta_lo']['beta'].mean()
        sigma_lo = df_est[df_est['bound'] == 'beta_lo']['beta'].std()
        x = np.linspace(mu_lo - 3*sigma_lo, mu_lo + 3*sigma_lo, 100)
        plt.plot(x, norm.pdf(x, mu_lo, sigma_lo), alpha=0.5, color='blue', linestyle='dashed', label='')

        # Add a legend but only for the first two entries
        plt.legend()

        # Move legend outside of plot region
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        # Get n and tol from dataframe
        n = df_est['n'].iloc[0]
        tol = df_est['tol'].iloc[0]
        r = df_est['r'].iloc[0]

        # Add N and tol as a note in the plot
        plt.text(0.95, 1, 'N = ' + str(n) + ',' + 'tol = ' + str(tol) + ',' + 'R = ' + str(r),
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=plt.gca().transAxes)
        
        # Also add the true mean, and estimated mean and sd as note in the plot
        plt.text(0.35, 0.8, 'True Mean = ' + str(round(beta_lo_true, 3)) + ', \n' + 'Estimated Mean = ' + str(round(mu_lo, 3)) + ',\n' + 'Estimated SD = ' + str(round(sigma_lo, 3)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=8,
                transform=plt.gca().transAxes)


        # Save graph and add n, r, kappa_n to the filename
        plt.savefig(dir_save + "/histo_beta_lo" + str(n) + "_" + str(r) + "_" + str(tol) + ".png", dpi=300)
        plt.show()
        

        # Make separate plots for beta_lo and beta_hi
        # Create a single plot overlaying two histograms, one for beta_lo, one for beta_hi
        plt.hist(df_est[df_est['bound'] == 'beta_hi']['beta'], color = 'red', bins='auto', density=True, alpha=0.5, label='Estimated Upper Bound')
        plt.axvline(x=beta_hi_true, alpha = 0.5, color='red', label='')
        plt.axvline(x=df_est[df_est['bound'] == 'beta_hi']['beta'].mean(), alpha = 0.5, color='red', linestyle='dashed', label='')

        mu_hi = df_est[df_est['bound'] == 'beta_hi']['beta'].mean()
        sigma_hi = df_est[df_est['bound'] == 'beta_hi']['beta'].std()
        x = np.linspace(mu_hi - 3*sigma_hi, mu_hi + 3*sigma_hi, 100)
        plt.plot(x, norm.pdf(x, mu_hi, sigma_hi), alpha=0.5, color='red', linestyle='dashed', label='')

        # Add a legend but only for the first two entries
        plt.legend()

        # Move legend outside of plot region
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        # Get n and tol from dataframe
        n = df_est['n'].iloc[0]
        tol = df_est['tol'].iloc[0]
        r = df_est['r'].iloc[0]

        # Add N and tol as a note in the plot
        plt.text(0.95, 1, 'N = ' + str(n) + ',' + 'tol = ' + str(tol) + ',' + 'R = ' + str(r),
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=plt.gca().transAxes)

        # Also add the true mean, and estimated mean and sd as note in the plot
        plt.text(0.35, 0.8, 'True Mean = ' + str(round(beta_hi_true,3)) + ',\n' + 'Estimated Mean = ' + str(round(mu_hi, 3)) + ',\n' + 'Estimated SD = ' + str(round(sigma_hi, 3)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=8,
                transform=plt.gca().transAxes)
        
        # Save graph and add n, r, kappa_n to the filename
        plt.savefig(dir_save + "/histo_beta_hi" + str(n) + "_" + str(r) + "_" + str(tol) + ".png", dpi=300)

        plt.show()