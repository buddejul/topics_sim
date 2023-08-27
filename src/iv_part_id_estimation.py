# Function running all code
from funcs import *
from estimation import *

def iv_part_id_estimation(
            y,
            z,
            d,
            target, 
            identif,
            basis, 
            shape=None,
            k0=None, 
            k1=None, 
            u_lo_target = None, 
            u_hi_target = None,
            u_lo_id = None, 
            u_hi_id = None,
            dz_cross = None,
            analyt_int = False,
            tol = None,
            quiet = False,
            return_gamma=False,
            u_part_fix=False,
            u_part_fix_tol=None
            ):
    
    """
    Arguments:
        - y (np.array): the outcome variable
        - z (np.array): the instrument(s)
        - d (np.array): the treatment variable
        - target: the target paramenter 
        - identif: the set of identified parameters
        - shape: shape restrictions on MTR functions
        - basis: the basis function to approximate M 
        - k0=None: the number of basis functions to approximate m0 
        - k1=None: the number of basis functions to approximate m1
        - u_lo_target = None: the lower bound of late if target is late
        - u_hi_target = None: the upper bound of late if target is late
        - u_lo_id = None: lower bounds of late if identified is late
        - u_hi_id = None: upper bounds of late if identified is late
        - dz_cross = None: the cross moment of the instrument(s)
        - analyt_int = False: whether to use analytical integration or not
        - tol: the tolerance for the estimatino problem
        - quiet = False: whether to print the ampl messages or not
    
    Returns:
        - results_min: the results of the minimization problem
        - results_max: the results of the maximization problem

    """

    if tol is None:
        raise ValueError("Need to specify tolerance for estimation problem")
    
    # if u_part_fix is True and u_part_fix_tol is None:
    #     raise ValueError("Need to specify tolerance for fixing partition")

    # If u_lo_target > u_hi_target, reverse them
    if u_lo_target > u_hi_target:
        u_lo_target, u_hi_target = u_hi_target, u_lo_target

    if u_lo_target > u_hi_target:
        raise ValueError("u_lo_target > u_hi_target")

    # Get number of each type of identified parameter
    n_iv_slope = len([i for i in identif if "iv_slope" in i])
    n_ols_slope = len([i for i in identif if "ols_slope" in i])
    n_cross = len([i for i in identif if "cross" in i])

    if n_iv_slope + n_ols_slope > 0 & n_cross > 0:
        raise ValueError("Cannot have both cross moments and other moments")

    # Compute relevant moments, p-score, and get support of z
    ez, ed, var_d, cov_dz = estimate_moments(y, z, d)
    p = estimate_prop_z(z, d)
    supp_z = np.unique(z)
    supp_z.sort() # make sure the propensity is ordered by the support of z
    f_z = estimate_f_z(z)
    z_p = p[np.searchsorted(supp_z, z)] # vector of propensity scores (length N)

    if u_part_fix is True:
        # FIXME only for very special setting of LATE(0.35, ..) target
        u_lo_target = p[0]

        

    # TODO Check this creates a new object (should work however, no assignment here)
    u_part = np.append(p, [u_lo_target, u_hi_target]) # add target cutoffs to partition
    u_part = np.unique(np.sort(u_part)) # make sure partition is ordered and remove duplicates
    u_part = np.insert(u_part, 0, 0) # add 0 to partition at beginning
    u_part = np.append(u_part, 1) # add 1 to partition

    # Remove close elements
    # if u_part_fix is True:
    #     differences = np.diff(u_part)
    #     indices_to_keep = np.insert(differences > u_part_fix_tol, 0, True)
    #     u_part = u_part[indices_to_keep]
        
    #     # FIXME this only works for our special setting
    #     u_lo_target = u_part[1]
    
    # TODO put function here that allows for target as functions of estimated
    # propensity score

    # Compute identified estimand
    val_estimands = np.zeros(len(identif))
    iv_df = []

    if n_iv_slope == 1:
        def s(d,z): return s_iv_slope(z, ez=ez, cov_dz=cov_dz)
        val_estimands[0] = estimate_beta(y, z, d, s)

    # TODO implement multiple instrumentss
   
    if n_ols_slope == 1:
        def s(d,z): return s_ols_slope(d, ed=ed, var_d=var_d)
        val_estimands[n_late + n_iv_slope] = compute_estimand_dgp(
            "ols_slope",
            m0_dgp,
            m1_dgp,
            supp_z = supp_z[0 + n_iv_slope],
            prop_z = prop_z[0 + n_iv_slope],
            f_z = f_z[0 + n_iv_slope])
    
    # TODO implement multiple OLS
    
    if n_cross == 1:
        def s(d,z): return s_cross(d, z, dz_cross=dz_cross[0])
        val_estimands[0] = estimate_beta(y, z, d, s)

    elif n_cross > 1:
        for i in range(n_cross):
            def s(d,z): return s_cross(d, z, dz_cross=dz_cross[i])
            val_estimands[i] = estimate_beta(y, z, d, s)
            
    for i,j in enumerate(val_estimands):
        df = pd.DataFrame(
        [
            ("val_identif_" + str(i), j)
        ],
        columns = ["IDENTIF_" + str(i), "val_identif_" + str(i)]
        ).set_index("IDENTIF_" + str(i))

        iv_df.append(df)

    # Compute gamma dataframe (target)
    # need to do this based on estimated moments (if not late)
    # TODO could this be reason for the bug? --> but we supply prop_z = p
    # which is the estimated moment so shouldn't be a problem?
    gamma_df = compute_gamma_df(
        target, 
        basis, 
        k0, 
        k1, 
        u_part, 
        u_lo_target, 
        u_hi_target, 
        supp_z=supp_z, 
        prop_z=p, 
        f_z=f_z, 
        analyt_int = analyt_int)
    
    # # Compute gamma dataframe (identif)
    gamma_id_df = []
    
   # TODO implement late

    if n_iv_slope == 1:
        df = compute_gamma_df_estimation(y, z, d, "iv_slope", "cs")

        df = df.rename(columns={"gamma": "gamma_ident_" + str(0)})
        gamma_id_df.append(df)
    
    if n_ols_slope == 1:
        df = compute_gamma_df_estimation(y, z, d, "ols_slope", "cs")

        df = df.rename(columns={"gamma": "gamma_ident_" + str(n_iv_slope)})
        gamma_id_df.append(df)
    
    elif n_cross == 1:
        df = compute_gamma_df_estimation(y, z, d, "cross", "cs", dz_cross=dz_cross[0])

        df = df.rename(columns={"gamma": "gamma_ident_" + str(n_iv_slope + n_ols_slope)})
        gamma_id_df.append(df)

    elif n_cross > 1:
        for i in range(n_cross):
                
                df = compute_gamma_df_estimation(y, z, d, "cross", "cs", dz_cross=dz_cross[i], u_part=u_part)
        
                df = df.rename(columns={"gamma": "gamma_ident_" + str(i + n_iv_slope + n_ols_slope)})
                gamma_id_df.append(df)

    # Now combine all gamma dataframes
    gamma_id_df = pd.concat(gamma_id_df, axis=1)


    # First we solve for the problem on the RHS of the constraint
    code_rhs_constraint = ampl_code_rhs_constraint(identif)
    inf_rhs = ampl_eval(code_rhs_constraint, None, gamma_id_df, iv_df, identif, quiet)

    code_max = ampl_code_estimation("maximize", identif, inf = inf_rhs[0], tol = tol)
    if quiet is False: print(code_max)
    results_max = ampl_eval(code_max, gamma_df, gamma_id_df, iv_df, identif, quiet)
    
    code_min = ampl_code_estimation("minimize", identif, inf = inf_rhs[0], tol = tol)
    if quiet is False: print(code_min)
    results_min = ampl_eval(code_min, gamma_df, gamma_id_df, iv_df, identif, quiet)

    if return_gamma is False:
        return results_max, results_min
    
    else:
        return results_max, results_min, gamma_df, gamma_id_df, iv_df, p, val_estimands
