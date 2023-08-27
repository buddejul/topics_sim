# Function that creates the main figure of the paper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from iv_part_id import *

def simulation_sharp_bounds(dir_results, dir_save, show=True, tol=None, n=None):
    """
    Args:
        tol (float): tolerance parameter
        dir_results (str): path to the folder containing the results
        dir_save (str): path to the folder where to save the figure
    Returns:
        None
    """

    # Define parameters
    target = "late"
    identif = ["cross", "cross", "cross", "cross", "cross", "cross"]
    basis = "cs"
    u_lo_target = 0.35

    f_z = [np.array([0.5, 0.4, 0.1])]
    supp_z = [np.array([0, 1, 2])]
    prop_z = [np.array([0.35, 0.6, 0.7])]

    dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def u_part(u_hi_target, prop_z):
        up = prop_z.copy()
        up.append(np.array([u_lo_target, u_hi_target]))
        up.append(np.array([0, 1]))
        up = np.unique(np.concatenate(up))
        up = np.sort(up)
        return up

    u_hi_targetlist = np.arange(0.4, 1, 0.025)

    # Get a list of u_part for each u_hi_target
    u_partlist = [u_part(u_hi_target, prop_z) for u_hi_target in u_hi_targetlist]

    # Combine u_hi_targetlist and u_partlist into a list of tuples
    u_list = list(zip(u_hi_targetlist, u_partlist))


    # Step 1: Get different bounds by target parameter
    def func(u_hi_target, u_part): 
        return iv_part_id(
            target, 
            identif,
            basis, 
            m0_dgp,
            m1_dgp,
            shape=None,
            k0=None, 
            k1=None, 
            u_part = u_part, 
            u_lo_target = u_lo_target, 
            u_hi_target = u_hi_target,
            supp_z = supp_z,
            prop_z = prop_z,
            f_z = f_z,
            dz_cross = dz_cross,
            analyt_int = True,
            quiet = False
        )
    
    out = [func(u_hi_target, u_part) for u_hi_target, u_part in u_list]

    # Now extract the bounds from the out
    beta_lo = [out[i][0][0] for i in range(len(out))]
    beta_hi = [out[i][1][0] for i in range(len(out))]

    # Put beta_hi and beta_lo into a dataframe
    df_true = pd.DataFrame({'beta_lo': beta_lo, 'beta_hi': beta_hi})
    # Add u_hi_targetlist to the dataframe
    df_true['u_hi_target'] = u_hi_targetlist
    df_true['u_lo_target'] = u_lo_target
    df_true

    # Step 2: Read in the estimates
    # For each element in u_hi_targetlist read the corresponding file in results
    # and extract the estimates

    # Read in all numpy elements in the folder results
    # Get all filenames in the folder results
    files = os.listdir(dir_results)

    # Get al lfilenames of type .npy
    files = [file for file in files if file[-4:] == ".npy"]

    # Read in all files in the folder results
    df_est = [read_results(dir_results, file) for file in files]

    # Put all dataframes into one dataframe
    df_est = pd.concat(df_est)
    df_est.head()

    if tol is not None:
        # Only keep df_est with tol = tol
        df_est = df_est[df_est['tol'] == tol]

    if n is not None:
        # Only keep df_est with N = N
        df_est = df_est[df_est['n'] == n]

    # Compute means by bound and u_hi_target
    df_est_mean = df_est.groupby(['bound', 'u_hi_target']).mean()

    # Compute standard deviations by bound and u_hi_target
    df_est_std = df_est.groupby(['bound', 'u_hi_target']).std()

    # Define the data for the violin plots
    fig, ax = plt.subplots(figsize=(16, 10))

    u_hi_targets = df_est['u_hi_target'].unique()

    colors = ['blue', 'green']

    for i, u_hi_target in enumerate(u_hi_targets):
        data_lo = df_est[(df_est['bound'] == 'beta_lo') & (df_est['u_hi_target'] == u_hi_target)]['beta']   
        data_hi = df_est[(df_est['bound'] == 'beta_hi') & (df_est['u_hi_target'] == u_hi_target)]['beta']

        # Check if the arrays are empty
        if len(data_lo) == 0 or len(data_hi) == 0:
            print(f'Error: One or more arrays are empty for u_hi_target={u_hi_target}')
        else:
            # Create the violin plots
            plots = ax.violinplot([data_lo, data_hi], positions=[u_hi_target, u_hi_target], 
                                showmeans=False, showmedians=False,
                                showextrema=False,
                                widths=0.1)

            # Set the color of the violin patches
            for pc, color in zip(plots['bodies'], colors):
                pc.set_facecolor(color)

    # Set the x-axis labels and title
    ax.set_xlabel('u_hi_target')
    ax.set_ylabel('beta')
    ax.set_title('Distribution of beta for beta_hi and beta_lo; target LATE(0.35, u_hi_target).')

    # Set the x-tick labels to match the x-axis values
    ax.set_xticks(u_hi_targets)
    rounded_u_hi_targets = [round(x, 2) for x in u_hi_targets]
    ax.set_xticklabels(rounded_u_hi_targets)

    # Set the y-axis limits
    ax.set_ylim([-1, 1])

    # Rotate the y-axis labels by 90 degrees
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

    # Now add the true bounds
    ax.plot(df_true['u_hi_target'], df_true['beta_lo'],
            label='True Lower Bound', color = 'blue')

    ax.plot(df_true['u_hi_target'], df_true['beta_hi'],
            label='True Upper Bound', color = 'green')

    # Add estimated means
    ax.plot(df_est_mean.loc['beta_lo'].index.values,
            df_est_mean.loc['beta_lo']['beta'],
            label='Mean Estimated Lower Bound', color = 'blue', linestyle='dashed')

    ax.plot(df_est_mean.loc['beta_hi'].index.values,
            df_est_mean.loc['beta_hi']['beta'],
            label='Mean Estimated Upper Bound', color = 'green', linestyle='dashed')

    # Add a legend
    ax.legend()

    # Get n, r, tol from df_est
    if n is None:
        n = df_est['n'].unique()[0]
    r = df_est['r'].unique()[0]
    if tol is None:
        tol = df_est['tol'].unique()[0]

    # Write a note in the plot in the bottom right corner
    plt.text(0.95, 0.05, 'n = ' + str(n) + '\n' + 'r = ' + str(r) + '\n' + 'kappa_n = ' + str(tol),
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=plt.gca().transAxes)

    ax.set_xlim([min(u_hi_targets), max(u_hi_targets+0.05)])

    # Save graph and add n, r, kappa_n to the filename
    plt.savefig(dir_save + "/simulation_sharp_bounds_" + str(n) + "_" + str(r) + "_" + str(tol) + ".png", dpi=300)

    if show:
        plt.show()

def read_results(dir, file):        
    out = np.load(dir + file)

    # Extract parameters from file
    # Remove .npy from filename
    file = file[:-4]
    file = file.split("_")
    bound = file[0] + "_" + file[1]
    n = int(file[2][1:])
    r = int(file[3][1:])
    tol = float(file[4][3:])
    u_lo_target = float(file[5][2:])
    u_hi_target = float(file[6][2:])

    # Put into dataframe
    df_est = pd.DataFrame({'bound': bound, 'n': n, 'r': r, 'tol': tol, 'u_lo_target': u_lo_target, 'u_hi_target': u_hi_target, 'beta': out})
    return df_est


def simulation_sharp_bounds_by_n(dir_results, dir_save, show=True, tol=None):
    """
    Args:
        tol (float): tolerance parameter
        dir_results (str): path to the folder containing the results
        dir_save (str): path to the folder where to save the figure
    Returns:
        None
    """

    # Define parameters
    target = "late"
    identif = ["cross", "cross", "cross", "cross", "cross", "cross"]
    basis = "cs"
    u_lo_target = 0.35

    f_z = [np.array([0.5, 0.4, 0.1])]
    supp_z = [np.array([0, 1, 2])]
    prop_z = [np.array([0.35, 0.6, 0.7])]

    dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def u_part(u_hi_target, prop_z):
        up = prop_z.copy()
        up.append(np.array([u_lo_target, u_hi_target]))
        up.append(np.array([0, 1]))
        up = np.unique(np.concatenate(up))
        up = np.sort(up)
        return up

    u_hi_targetlist = np.arange(0.4, 1, 0.025)

    # Get a list of u_part for each u_hi_target
    u_partlist = [u_part(u_hi_target, prop_z) for u_hi_target in u_hi_targetlist]

    # Combine u_hi_targetlist and u_partlist into a list of tuples
    u_list = list(zip(u_hi_targetlist, u_partlist))


    # Step 1: Get different bounds by target parameter
    def func(u_hi_target, u_part): 
        return iv_part_id(
            target, 
            identif,
            basis, 
            m0_dgp,
            m1_dgp,
            shape=None,
            k0=None, 
            k1=None, 
            u_part = u_part, 
            u_lo_target = u_lo_target, 
            u_hi_target = u_hi_target,
            supp_z = supp_z,
            prop_z = prop_z,
            f_z = f_z,
            dz_cross = dz_cross,
            analyt_int = True,
            quiet = False
        )
    
    out = [func(u_hi_target, u_part) for u_hi_target, u_part in u_list]

    # Now extract the bounds from the out
    beta_lo = [out[i][0][0] for i in range(len(out))]
    beta_hi = [out[i][1][0] for i in range(len(out))]

    # Put beta_hi and beta_lo into a dataframe
    df_true = pd.DataFrame({'beta_lo': beta_lo, 'beta_hi': beta_hi})
    # Add u_hi_targetlist to the dataframe
    df_true['u_hi_target'] = u_hi_targetlist
    df_true['u_lo_target'] = u_lo_target
    df_true

    # Step 2: Read in the estimates
    # For each element in u_hi_targetlist read the corresponding file in results
    # and extract the estimates

    # Read in all numpy elements in the folder results
    # Get all filenames in the folder results
    files = os.listdir(dir_results)

    # Get al lfilenames of type .npy
    files = [file for file in files if file[-4:] == ".npy"]

    # Read in all files in the folder results
    df_est = [read_results(dir_results, file) for file in files]

    # Put all dataframes into one dataframe
    df_est = pd.concat(df_est)
    df_est.head()

    if tol is not None:
        # Only keep df_est with tol = tol
        df_est = df_est[df_est['tol'] == tol]

    # Compute means by bound, u_hi_target, and n
    df_est_mean = df_est.groupby(['bound', 'u_hi_target', 'n']).mean()

    # Compute standard deviations
    df_est_std = df_est.groupby(['bound', 'u_hi_target', 'n']).std()

    # Define the data for the violin plots
    fig, ax = plt.subplots(figsize=(16, 10))

    # Set the x-axis labels and title
    ax.set_xlabel('u_hi_target')
    ax.set_ylabel('beta')
    ax.set_title('Means of beta for beta_hi and beta_lo; target LATE(0.35, u_hi_target).')

    # Set the y-axis limits
    ax.set_ylim([-1, 1])

    # Now add the true bounds
    ax.plot(df_true['u_hi_target'], df_true['beta_lo'],
            label='True Lower Bound', color = 'blue')

    ax.plot(df_true['u_hi_target'], df_true['beta_hi'],
            label='True Upper Bound', color = 'green')

    # Add estimated means separately for each n
    for i, n in enumerate(df_est['n'].unique()):
        subset = df_est_mean.loc[df_est_mean.index.get_level_values('n') == n]
        u_hi_target_vals = subset.index.get_level_values('u_hi_target').unique()

        if i == 0: linestyle = '--'
        if i == 1: linestyle = '-.'
        if i == 2: linestyle = ':'

        ax.plot(u_hi_target_vals,
                subset.loc['beta_lo']['beta'],
                label='Mean Estimated Lower Bound, n=' + str(n), color = 'blue', linestyle=linestyle)

        ax.plot(u_hi_target_vals,
                subset.loc['beta_hi']['beta'],
                label='Mean Estimated Upper Bound, n=' + str(n), color = 'green', linestyle=linestyle)
    
    # Add standard deviations separately for each n via error bars
    for n in df_est['n'].unique():
        subset = df_est_std.loc[df_est_std.index.get_level_values('n') == n]
        subset_means = df_est_mean.loc[df_est_mean.index.get_level_values('n') == n]
        u_hi_target_vals = subset.index.get_level_values('u_hi_target').unique()
        ax.errorbar(u_hi_target_vals,
                    subset_means.loc['beta_lo']['beta'],
                    yerr=subset.loc['beta_lo']['beta'],
                    label='Standard Deviation Estimated Lower Bound, n=' + str(n), color = 'blue', linestyle='dashed')

        ax.errorbar(u_hi_target_vals,
                    subset_means.loc['beta_hi']['beta'],
                    yerr=subset.loc['beta_hi']['beta'],
                    label='Standard Deviation Estimated Upper Bound, n=' + str(n), color = 'green', linestyle='dashed')
     

    # Add a legend
    ax.legend()

    # Get n, r, tol from df_est
    if n is None:
        n = df_est['n'].unique()[0]
    r = df_est['r'].unique()[0]
    if tol is None:
        tol = df_est['tol'].unique()[0]

    # Write a note in the plot in the bottom right corner
    plt.text(0.95, 0.05, 'r = ' + str(r) + '\n' + 'kappa_n = ' + str(tol),
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=plt.gca().transAxes)

    # Save graph and add n, r, kappa_n to the filename
    plt.savefig(dir_save + "/simulation_sharp_bounds_by_n_" + str(r) + "_" + str(tol) + ".png", dpi=300)

    if show:
        plt.show()

def read_results(dir, file):        
    out = np.load(dir + file)

    # Extract parameters from file
    # Remove .npy from filename
    file = file[:-4]
    file = file.split("_")
    bound = file[0] + "_" + file[1]
    n = int(file[2][1:])
    r = int(file[3][1:])
    tol = float(file[4][3:])
    u_lo_target = float(file[5][2:])
    u_hi_target = float(file[6][2:])

    # Put into dataframe
    df_est = pd.DataFrame({'bound': bound, 'n': n, 'r': r, 'tol': tol, 'u_lo_target': u_lo_target, 'u_hi_target': u_hi_target, 'beta': out})
    return df_est