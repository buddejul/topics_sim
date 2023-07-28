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
            quiet = False
            ):

    # Compute identified estimand
    if identif == "iv_slope":
        val_id_estimand = compute_estimand_dgp(
            identif,
            m0_dgp,
            m1_dgp,
            supp_z = supp_z,
            prop_z = prop_z,
            f_z = f_z)
        
    elif identif == "late":
        val_id_estimand = compute_estimand_dgp(
            identif,
            m0_dgp,
            m1_dgp,
            u_lo = u_lo_id,
            u_hi = u_hi_id)

    iv_df = gen_identif_df(val_id_estimand)
      
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
    gamma_id_df = compute_gamma_df(
        identif,
        basis,
        k0,
        k1,
        u_part,
        u_lo_id,
        u_hi_id,
        supp_z,
        prop_z,
        f_z)
    
    gamma_id_df = gamma_id_df.rename(columns={"gamma": "gamma_ident"})

    code = ampl_code("minimize")
    results_min = ampl_eval(code, gamma_df, gamma_id_df, iv_df, quiet)

    code = ampl_code("maximize")
    results_max = ampl_eval(code, gamma_df, gamma_id_df, iv_df, quiet)

    return results_min, results_max
