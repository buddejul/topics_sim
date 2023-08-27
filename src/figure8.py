# Replicate Figure 8 from the paper

def plot_figure8(df_true, df_est, savepath=None):

    # Compute means by bound and u_hi_target
    df_est_mean = df_est.groupby(['bound', 'u_hi_target']).mean()
    df_est_mean

    # Compute standard deviations by bound and u_hi_target
    df_est_std = df_est.groupby(['bound', 'u_hi_target']).std()

    # Create a plot of beta_lo, beta_hi with u_target_hi on the x-axis
    plt.plot(df_true['u_hi_target'], df_true['beta_lo'], 
        label='True Lower Bound', color = 'blue')

    plt.plot(df_true['u_hi_target'], df_true['beta_hi'], 
        label='True Upper Bound', color = 'green')

    plt.plot(df_est_mean.loc['beta_lo']['beta'], 
        label='Mean Estimated Lower Bound', color = 'blue', linestyle='dashed')

    plt.plot(df_est_mean.loc['beta_hi']['beta'], 
        label='Mean Estimated Upper Bound', color = 'green', linestyle='dashed')

    # Add error bars around means using std
    plt.fill_between(df_est_mean.loc['beta_lo']['beta'].index,
                        df_est_mean.loc['beta_lo']['beta'] - df_est_std.loc['beta_lo']['beta'],
                        df_est_mean.loc['beta_lo']['beta'] + df_est_std.loc['beta_lo']['beta'],
                        alpha=0.2, color = 'blue')
    plt.fill_between(df_est_mean.loc['beta_hi']['beta'].index,
                        df_est_mean.loc['beta_hi']['beta'] - df_est_std.loc['beta_hi']['beta'],
                        df_est_mean.loc['beta_hi']['beta'] + df_est_std.loc['beta_hi']['beta'],
                        alpha=0.2, color = 'green')


    plt.xlabel('u_hi_target')
    plt.ylabel('')
    plt.legend()
    plt.title('True and Estimated Bounds for Late(0.35, u_hi_target)')

    # Get n, r, tol from df_est
    n = df_est['n'].unique()[0]
    r = df_est['r'].unique()[0]
    tol = df_est['tol'].unique()[0]

    # Write a note in the plot in the bottom right corner
    plt.text(0.95, 0.05, 'n = ' + str(n) + '\n' + 'r = ' + str(r) + '\n' + 'tol = ' + str(tol),
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=plt.gca().transAxes)
    
    if savepath is not None:
        plt.savefig(savepath + "/figure8_" + str(n) + "_" + str(r) + "_" + str(tol) + ".png", dpi=300)

    plt.show()

    return None