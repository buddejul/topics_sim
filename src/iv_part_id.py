# Function running all code
from funcs import *

def iv_part_id(
            target, 
            identif,
            basis, 
            m0_dgp,
            m1_dgp,
            k0=None, 
            k1=None, 
            u_part = None, 
            u_lo_target = None, 
            u_hi_target = None,
            u_lo_id = None, 
            u_hi_id = None,
            supp_z = None,
            prop_z = None,
            f_z = None,
            dz_cross = None,
            quiet = False
            ):
    
    """
    Arguments:
        - target: the target paramenter 
        - identif: the set of identified parameters
        - basis: the basis function to approximate M 
        - m0_dgp: the DGP for m0
        - m1_dgp: the DGP for m1
        - k0=None: the number of basis functions to approximate m0 
        - k1=None: the number of basis functions to approximate m1
        - u_part = None: the partition of u into bins with constant weights
        - u_lo_target = None: the lower bound of late if target is late
        - u_hi_target = None: the upper bound of late if target is late
        - u_lo_id = None: lower bounds of late if identified is late
        - u_hi_id = None: upper bounds of late if identified is late
        - supp_z = None: support of the instrument(s)
        - prop_z = None: propensity score as a function of the instrument(s)
        - f_z = None: support of the instrument(s)
        - dz_cross = None: the cross moment of the instrument(s)
        - quiet = False: whether to print the ampl messages or not
    
    Returns:
        - results_min: the results of the minimization problem
        - results_max: the results of the maximization problem

    """

    # Get number of each type of identified parameter
    n_late = len([i for i in identif if "late" in i])
    n_iv_slope = len([i for i in identif if "iv_slope" in i])
    n_ols_slope = len([i for i in identif if "ols_slope" in i])
    n_cross = len([i for i in identif if "cross" in i])

    if n_late + n_iv_slope + n_ols_slope > 0 & n_cross > 0:
        raise ValueError("Cannot have both cross moments and other moments")

    # Compute identified estimand
    val_estimands = np.zeros(len(identif))
    iv_df = []

    if n_late == 1:
        val_estimands[0] = compute_estimand_dgp(
            "late",
            m0_dgp,
            m1_dgp,
            u_lo = u_lo_id,
            u_hi = u_hi_id)
    elif n_late > 1:
        for i in range(n_late):
            val_estimands[i] = compute_estimand_dgp(
                "late",
                m0_dgp,
                m1_dgp,
                u_lo = u_lo_id[i],
                u_hi = u_hi_id[i])

    if n_iv_slope == 1:
        val_estimands[n_late] = compute_estimand_dgp(
            "iv_slope",
            m0_dgp,
            m1_dgp,
            supp_z = supp_z[0],
            prop_z = prop_z[0],
            f_z = f_z[0])
    elif n_iv_slope > 1:
        for i in range(n_iv_slope):
            val_estimands[i + n_late] = compute_estimand_dgp(
                "iv_slope",
                m0_dgp,
                m1_dgp,
                supp_z = supp_z[i],
                prop_z = prop_z[i],
                f_z = f_z[i])
   
    if n_ols_slope == 1:
        val_estimands[n_late + n_iv_slope] = compute_estimand_dgp(
            "ols_slope",
            m0_dgp,
            m1_dgp,
            supp_z = supp_z[0 + n_iv_slope],
            prop_z = prop_z[0 + n_iv_slope],
            f_z = f_z[0 + n_iv_slope])
    elif n_ols_slope > 1:
        for i in range(n_ols_slope):
            val_estimands[i + n_late + n_iv_slope] = compute_estimand_dgp(
                "ols_slope",
                m0_dgp,
                m1_dgp,
                supp_z = supp_z[i],
                prop_z = prop_z[i],
                f_z = f_z[i])
    
    if n_cross == 1:
        val_estimands[0] = compute_estimand_dgp(
            "cross",
            m0_dgp,
            m1_dgp,
            supp_z = supp_z[0],
            prop_z = prop_z[0],
            f_z = f_z[0],
            dz_cross = dz_cross[0])
    elif n_cross > 1:
        for i in range(n_cross):
            val_estimands[i] = compute_estimand_dgp(
                "cross",
                m0_dgp,
                m1_dgp,
                supp_z = supp_z[0],
                prop_z = prop_z[0],
                f_z = f_z[0],
                dz_cross = dz_cross[i])
            
    for i,j in enumerate(val_estimands):
        df = pd.DataFrame(
        [
            ("val_identif_" + str(i), j)
        ],
        columns = ["IDENTIF_" + str(i), "val_identif_" + str(i)]
        ).set_index("IDENTIF_" + str(i))

        iv_df.append(df)

    # Compute gamma dataframe (target)
    gamma_df = compute_gamma_df(
        target, 
        basis, 
        k0, 
        k1, 
        u_part, 
        u_lo_target, 
        u_hi_target, 
        supp_z, 
        prop_z, 
        f_z)
    
    # Compute gamma dataframe (identif)
    gamma_id_df = []
    
    if n_late == 1:
        df = compute_gamma_df(
            "late",
            basis=basis,
            k0=k0,
            k1=k1,
            u_part=u_part,
            u_lo=u_lo_id,
            u_hi=u_hi_id,
            )

        df = df.rename(columns={"gamma": "gamma_ident_" + str(0)})
        gamma_id_df.append(df)
    elif n_late > 1:
        for i in range(n_late):

            df = compute_gamma_df(
                "late",
                basis=basis,
                k0=k0,
                k1=k1,
                u_part=u_part,
                u_lo=u_lo_id[i],
                u_hi=u_hi_id[i],
                )

            df = df.rename(columns={"gamma": "gamma_ident_" + str(i)})
            gamma_id_df.append(df)

    if n_iv_slope == 1:
        df = compute_gamma_df(
            "iv_slope",
            basis=basis,
            k0=k0,
            k1=k1,
            u_part=u_part,
            supp_z=supp_z[0],
            prop_z=prop_z[0],
            f_z=f_z[0]
            )

        df = df.rename(columns={"gamma": "gamma_ident_" + str(n_late)})
        gamma_id_df.append(df)
    
    elif n_iv_slope > 1:
        for i in range(n_iv_slope):
                
                df = compute_gamma_df(
                    "iv_slope",
                    basis=basis,
                    k0=k0,
                    k1=k1,
                    u_part=u_part,
                    supp_z=supp_z[i],
                    prop_z=prop_z[i],
                    f_z=f_z[i]
                    )
        
                df = df.rename(columns={"gamma": "gamma_ident_" + str(i + n_late)})
                gamma_id_df.append(df)

    if n_ols_slope == 1:
        df = compute_gamma_df(
            "ols_slope",
            basis=basis,
            k0=k0,
            k1=k1,
            u_part=u_part,
            supp_z=supp_z[0 + n_iv_slope],
            prop_z=prop_z[0 + n_iv_slope],
            f_z=f_z[0 + n_iv_slope]
            )

        df = df.rename(columns={"gamma": "gamma_ident_" + str(n_late + n_iv_slope)})
        gamma_id_df.append(df)
    
    elif n_ols_slope > 1:
        for i in range(n_ols_slope):
                
                df = compute_gamma_df(
                    "ols_slope",
                    basis=basis,
                    k0=k0,
                    k1=k1,
                    u_part=u_part,
                    supp_z=supp_z[i],
                    prop_z=prop_z[i],
                    f_z=f_z[i]
                    )
        
                df = df.rename(columns={"gamma": "gamma_ident_" + str(i + n_late + n_iv_slope)})
                gamma_id_df.append(df)

    elif n_cross == 1:
        df = compute_gamma_df(
            "cross",
            basis=basis,
            k0=k0,
            k1=k1,
            u_part=u_part,
            supp_z=supp_z[0],
            prop_z=prop_z[0],
            f_z=f_z[0],
            dz_cross=dz_cross[0]
            )

        df = df.rename(columns={"gamma": "gamma_ident_" + str(n_late + n_iv_slope + n_ols_slope)})
        gamma_id_df.append(df)

    elif n_cross > 1:
        for i in range(n_cross):
                
                df = compute_gamma_df(
                    "cross",
                    basis=basis,
                    k0=k0,
                    k1=k1,
                    u_part=u_part,
                    supp_z=supp_z[0],
                    prop_z=prop_z[0],
                    f_z=f_z[0],
                    dz_cross=dz_cross[i]
                    )
        
                df = df.rename(columns={"gamma": "gamma_ident_" + str(i + n_late + n_iv_slope + n_ols_slope)})
                gamma_id_df.append(df)

    # Now combine all gamma dataframes
    gamma_id_df = pd.concat(gamma_id_df, axis=1)

    code = ampl_code("minimize", identif)
    print(code)
    results_min = ampl_eval(code, gamma_df, gamma_id_df, iv_df, identif, quiet)

    code = ampl_code("maximize", identif)
    print(code)
    results_max = ampl_eval(code, gamma_df, gamma_id_df, iv_df, identif, quiet)

    return results_min, results_max
