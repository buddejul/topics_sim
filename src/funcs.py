import math
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from amplpy import AMPL

# Define basis functions
def cs_basis(u_lo, u_hi, u):
    """Constant spline basis function (no covariates)"""
    if u_lo <= u < u_hi: return 1
    else: return 0

def bern_bas(n, v, x):
    """Bernstein polynomial basis of degree n and index v at point x"""
    return math.comb(n, v) * x ** v * (1 - x) ** (n - v)

# Define s functions
def s_iv_slope(z, ez, cov_dz):
    """IV-like specification s(d,z): IV slope
    Args:
        z (np.int): value of the instrument
        ez (np.float): expected value of the instrument
        cov_dz (np.float): covariance between treatment and instrument
    """
    return (z - ez) / cov_dz

def s_ols_slope(d, ed, var_d):
    """OLS-like specification s(d,z): OLS slope
    Args:
        d (np.int): value of the treatment
        ed (np.float): expected value of the treatment
        var_d (np.float): variance of the treatment
    """
    return (d - ed) / var_d

def s_late(d, u, u_lo, u_hi):
    """IV-like specification s(d,z): late
    """
    # Return 1 divided by u_hi - u_lo if u_lo < u < u_hi, 0 otherwise
    if u_lo < u < u_hi: w = 1 / (u_hi - u_lo)
    else: w = 0

    if d == 1: return w
    else: return -w

def s_cross(d, z, dz_cross):
    """ IV_like specification s(d,z): Cross-moment d_spec * z_spec
    """
    if d == dz_cross[0] and z == dz_cross[1]: return 1
    else: return 0


# Gamma* linear maps
def gamma_star(
        md,
        d,
        estimand,
        u_lo=0,
        u_hi=1,
        supp_z=None,
        prop_z=None,
        f_z=None,
        dz_cross=None,
        analyt_int=False,
        u_part = None,
        u_part_lo = None,
        u_part_hi = None):
    
    """ Compute gamma* for a given MTR function and estimand
    Args:
        md (function): MTR function
        d (np.int): value of the treatment
        estimand (str): the estimand to compute
        u_lo (float): lower bound of late target
        u_hi (float): upper bound of late target
        supp_z (np.array): support of the instrument
        prop_z (np.array): propensity given the instrument
        f_z (np.array): probability mass function of the instrument
        dz_cross (list): list of tuples of the form (d_spec, z_spec) for cross-moment
        analyt_int (Boolean): Whether to integate manually or use analytic results
    """

    if estimand not in ["iv_slope", "late", "ols_slope", "cross"]:
        raise ValueError("estimand must be either 'iv_slope', 'late', 'ols_slope' or 'cross'")

    if estimand in ["iv_slope", "ols_slope"]:
        if supp_z is None or prop_z is None or f_z is None:
            raise ValueError("supp_z, prop_z, and f_z must be specified for iv_slope or ols_slope")
        
    if estimand == "cross":
        if dz_cross is None:
            raise ValueError("dz_cross must be specified for cross-moment")
    
    if analyt_int == True and u_part is None:
            raise ValueError("u_part must be specified for cs basis")

    if estimand == "late": 
        return integrate.quad(
            lambda u: md(u) * s_late(d, u, u_lo, u_hi), u_lo, u_hi)[0]
    
    # Do integration manually via scipy integrate
    if analyt_int == False: 
        if estimand == "iv_slope":
            ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)

            if d == 0:
                def func(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] < u: return md(u) * s_iv_slope(z, ez, cov_dz)
                    else : return 0

            if d == 1:
                def func(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] > u: return md(u) * s_iv_slope(z, ez, cov_dz)
                    else : return 0

            # Integrate func over u in [0,1] for every z in supp_z
            return np.sum([integrate.quad(func, 0, 1, args=(z,))[0] * f_z[i] 
                for i, z in enumerate(supp_z)])
        
        if estimand == "ols_slope":
            ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)
            var_d = ed * (1 - ed)

            if d == 0:
                # need to condition on z
                def func(u, z):
                    if prop_z[np.where(supp_z == z)[0][0]] < u: return md(u) * s_ols_slope(d, ed, var_d)
                    else : return 0

            if d == 1:
                def func(u, z):
                    if prop_z[np.where(supp_z == z)[0][0]] > u: return md(u) * s_ols_slope(d, ed, var_d)
                    else : return 0

            # Integrate func over u in [0,1] for every z in supp_z
            return np.sum([integrate.quad(func, 0, 1, args=(z,))[0] * f_z[i]
                for i, z in enumerate(supp_z)])
        
        if estimand == "cross":
            if d == 0:
                def func(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] < u: return md(u) * s_cross(d, z, dz_cross)
                    else: return 0
            
            if d == 1:
                def func(u, z): 
                    if prop_z[np.where(supp_z == z)[0][0]] >= u: return md(u) * s_cross(d, z, dz_cross)
                    else: return 0

            # Integrate func over u in [0,1] for every z in supp_z
            return np.sum([integrate.quad(func, 0, 1, args=(z,))[0] * f_z[i]
                for i, z in enumerate(supp_z)])

    # Use analytic results on constant spline basis
    if analyt_int == True:

        if estimand == "iv_slope":
            ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)

            if d == 0: 
                    return (u_part_hi - u_part_lo) * np.sum(
                            [f_z[j] * s_iv_slope(z, ez, cov_dz) * (prop_z[j] <= u_part_lo)
                            for j, z in enumerate(supp_z)])


            if d == 1:
                    return (u_part_hi - u_part_lo) * np.sum(
                            [f_z[j] * s_iv_slope(z, ez, cov_dz) * (prop_z[j] >= u_part_hi)
                            for j, z in enumerate(supp_z)])
            
        if estimand == "ols_slope":
            ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)
            var_d = ed * (1 - ed)

            if d == 0: 
                    return (u_part_hi - u_part_lo) * np.sum(
                            [f_z[j] * s_ols_slope(d, ed, var_d) * (prop_z[j] <= u_part_lo)
                            for j, z in enumerate(supp_z)])


            if d == 1:
                    return (u_part_hi - u_part_lo) * np.sum(
                            [f_z[j] * s_ols_slope(d, ed, var_d) * (prop_z[j] >= u_part_hi)
                            for j, z in enumerate(supp_z)])
            
        if estimand == "cross":
            ez, ed, edz, cov_dz = compute_moments(supp_z, f_z, prop_z)
            var_d = ed * (1 - ed)

            if d == 0: 
                    return (u_part_hi - u_part_lo) * np.sum(
                            [f_z[j] * s_cross(d, z, dz_cross) * (prop_z[j] <= u_part_lo)
                            for j, z in enumerate(supp_z)])


            if d == 1:
                    return (u_part_hi - u_part_lo) * np.sum(
                            [f_z[j] * s_cross(d, z, dz_cross) * (prop_z[j] >= u_part_hi)
                            for j, z in enumerate(supp_z)])
            
def compute_moments(supp_z, f_z, prop_z):
    """Calculate E[z], E[d], E[dz], Cov[d,z] for a discrete instrument z
    and binary d 
    Args:
        supp_z (np.array): support of the instrument
        f_z (np.array): probability mass function of the instrument
        prop_z (np.array): propensity score of the instrument
    """

    ez = np.sum(supp_z * f_z)
    ed = np.sum(prop_z * f_z)
    edz = np.sum(supp_z * prop_z * f_z)
    cov_dz = edz - ed * ez

    return ez, ed, edz, cov_dz

# DGP
def m0_dgp(u): 
    return 0.6 * bern_bas(2, 0, u) + 0.4 * bern_bas(2, 1, u) + 0.3 * bern_bas(2, 2, u)

def m1_dgp(u):
    return 0.75 * bern_bas(2, 0, u) + 0.5 * bern_bas(2, 1, u) + 0.25 * bern_bas(2, 2, u)

# Function computing identified parameters from dgp
def compute_estimand_dgp(estimand, m0, m1, u_lo = 0, u_hi = 1, 
                     supp_z = None, prop_z = None, f_z = None, dz_cross = None,
                     u_part = None, analyt_int = False):

    if estimand not in ["iv_slope", "late", "ols_slope", "cross"]:
        raise ValueError("estimand must be either 'iv_slope', 'late' 'ols_slope', or 'cross'")
    
    if estimand == "late":
        a = gamma_star(m0, 0, estimand, u_lo = u_lo, u_hi = u_hi, u_part = u_part, analyt_int = analyt_int)
        b = gamma_star(m1, 1, estimand, u_lo = u_lo, u_hi = u_hi, u_part = u_part, analyt_int = analyt_int)

    elif estimand == "iv_slope" or estimand == "ols_slope":
        a = gamma_star(m0, 0, estimand, 
        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
        u_part = u_part, analyt_int = analyt_int)
        
        b = gamma_star(m1, 1, estimand, 
        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
        u_part = u_part, analyt_int = analyt_int) 
        

    elif estimand == "cross":
        a = gamma_star(m0, 0, estimand,
        dz_cross = dz_cross,
        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
        u_part = u_part, analyt_int = analyt_int)

        b = gamma_star(m1, 1, estimand,
        dz_cross = dz_cross,
        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
        u_part = u_part, analyt_int = analyt_int)

    return a + b

# Function writing AMPL model code

# Function sending data to AMPL

def compute_gamma_df(target, basis, k0 = None, k1 = None, u_part = None, u_lo=None, u_hi=None,
                     supp_z = None, prop_z = None, f_z = None, dz_cross = None, analyt_int = False):
    """
    Compute gamma* evaluated at different basis functions
    Args:
        target (str): the target estimand
        k0 (int): degree of the polynomial for the d = 0 MTE
        k1 (int): degree of the polynomial for the d = 1 MTE
        basis (str): the basis function to use
        u_part (np.array): partition of [0,1] for cs basis function
        u_lo (float): lower bound of late target
        u_hi (float): upper bound of late target
        supp_z (np.array): support of the instrument
        prop_z (np.array): propensity given the instrument
        f_z (np.array): probability mass function of the instrument
        dz_cross (list): list of tuples of the form (d_spec, z_spec) for cross-moment

    Returns:
        gamma_df (pd.DataFrame): a dataframe of length k0 + k1
    """
    
    if target not in ["iv_slope", "late", "ols_slope", "cross"]:
        raise ValueError("target must be either 'iv_slope', 'late', 'ols_slope' or 'cross'")
    
    if basis not in ["bernstien", "cs"]:
        raise ValueError("basis must be either 'bernstein' or 'cs'")
    
    if target == "late" and (u_lo is None or u_hi is None):
        raise ValueError("u_lo and u_hi must be specified for late target")
    
    if basis == "cs" and u_part is None:
        raise ValueError("u_part must be specified for cs basis")
    
    if target == "cross" and dz_cross is None:
        raise ValueError("dz_cross must be specified for cross-moment")

    # Compute gamma* for d = 0

    if basis == "bernstein":
        gamma0 = np.zeros(k0)
        gamma1 = np.zeros(k1)

    if basis == "cs":
        gamma0 = np.zeros(len(u_part) - 1)
        gamma1 = np.zeros(len(u_part) - 1)

    if basis == "cs":
        for i, u in enumerate(u_part):
            if i < len(u_part) - 1:
                def func(u): return cs_basis(u_part[i], u_part[i+1], u)

                if target == "late":
                    gamma0[i] = gamma_star(func, 0, target, 
                        u_lo = u_lo, u_hi = u_hi, u_part = u_part, analyt_int = analyt_int,
                        u_part_lo = u_part[i], u_part_hi = u_part[i+1])
                    
                    gamma1[i] = gamma_star(func, 1, target, 
                        u_lo = u_lo, u_hi = u_hi, u_part = u_part, analyt_int = analyt_int,
                        u_part_lo = u_part[i], u_part_hi = u_part[i+1])

                elif target == "iv_slope" or target == "ols_slope":
                    gamma0[i] = gamma_star(func, 0, target, 
                        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
                        u_part = u_part, analyt_int = analyt_int,
                        u_part_lo = u_part[i], u_part_hi = u_part[i+1])
                    
                    gamma1[i] = gamma_star(func, 1, target, 
                        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
                        u_part = u_part, analyt_int = analyt_int,
                        u_part_lo = u_part[i], u_part_hi = u_part[i+1])
                    
                elif target == "cross":
                    gamma0[i] = gamma_star(func, 0, target, 
                        dz_cross = dz_cross,
                        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
                        u_part = u_part, analyt_int = analyt_int,
                        u_part_lo = u_part[i], u_part_hi = u_part[i+1])
                    
                    gamma1[i] = gamma_star(func, 1, target, 
                        dz_cross = dz_cross,
                        supp_z = supp_z, prop_z = prop_z, f_z = f_z, 
                        u_part = u_part, analyt_int = analyt_int,
                        u_part_lo = u_part[i], u_part_hi = u_part[i+1])
    

    # Generate column vector of names for d=0, d=1, and k1, k2
    if basis == "bernstein":
        d0 = ["theta0_" + str(i) for i in range(k0)]
        d1 = ["theta1_" + str(i) for i in range(k1)]
        d = d0 + d1

    if basis == "cs":
        d0 = ["theta0_" + str(i) for i in range(len(u_part) - 1)]
        d1 = ["theta1_" + str(i) for i in range(len(u_part) - 1)]
        d = d0 + d1

    # Concatenate d, theta0 + theta1
    gamma = np.concatenate((gamma0, gamma1))

    df = pd.DataFrame(gamma, d, columns=['gamma'])
    df.index.rename("THETA", inplace=True)

    return df

# Write a function that generates this AMPL code and runs it in ampl_eval
# Function that generates AMPL code
def ampl_code(min_or_max, identif, shape, u_part):
    """
    Generate AMPL code for the identified set
    Args:
        gamma_df (pd.DataFrame): a dataframe of length k0 + k1
        iv_df (pd.DataFrame): a dataframe of length k0 + k1
    Returns:
        ampl_code (str): AMPL code for the identified set
    """
    # Generate AMPL code for the identified set
    # Define sets
    ampl_code = "reset;\n"
    ampl_code += "set THETA;\n"
    for i in range(len(identif)):
        ampl_code += "set IDENTIF_" + str(i) + ";\n"

    # Define parameters
    ampl_code += "param gamma {THETA};\n"
    for i in range(len(identif)):
        ampl_code += "param gamma_ident_" + str(i) + " {THETA};\n"
        ampl_code += "param val_identif_" + str(i) + " {IDENTIF_" + str(i) + "};\n"

    # Choice variable (note this is equivalent to Y in [0,1] constraint, see MST 2018 Appendix)
    ampl_code += "var Theta_val {j in THETA} >= 0, <= 1;\n"

    # Objective function    
    ampl_code += min_or_max + " beta: sum {j in THETA} gamma[j] * Theta_val[j];\n"

    # Constraints
    for i in range(len(identif)):
        ampl_code += "subject to Identified_" + str(i) + " {i in IDENTIF_" + str(i) + "}:\n"
        ampl_code += "sum {j in THETA} gamma_ident_" + str(i) + "[j] * Theta_val[j] == val_identif_" + str(i) + "[i];\n"

    # Shape restrictions (for Bernstein polynomials or constant splines)
    if shape != None:
        if len(shape) == 1 and shape[0] == "decr":
            for j in range(len(u_part)-2):
                ampl_code += "subject to DecreasingConstraint_d0_" + str(j) + ":\n"
                ampl_code += "Theta_val['theta0_" + str(j) + "'] >= Theta_val['theta0_" + str(j+1) + "'];\n"

                ampl_code += "subject to DecreasingConstraint_d1_" + str(j) + ":\n"
                ampl_code += "Theta_val['theta1_" + str(j) + "'] >= Theta_val['theta1_" + str(j+1) + "'];\n"

            # ampl_code += "var Binary_var {j in 2..card(THETA)} binary";  # Binary variable for ordering

            # # Define constraints to enforce decreasing order
            # ampl_code += "subject to OrderConstraint {j in 2..card(THETA)}:"
            # ampl_code += "Theta_val[j-1] >= Theta_val[j] - Binary_var[j];"

            # # Set the binary variables consistently with the ordering
            # ampl_code += "subject to BinarySettingConstraint {j in 2..card(THETA)}:"
            # ampl_code += " Binary_var[j] <= Binary_var[j-1];"

    if shape != None:
        if len(shape) == 1 and shape[0] == "incr":
            ampl_code += "subject to Increasing {j in 1..card(THETA)-1}:\n"
            ampl_code += "Theta_val[j] <= Theta_val[j+1];\n"
        
    return ampl_code

def combine_gamma_df(gamma_df, gamma_ident_df):
    """Combine gamma_df and gamma_ident_df into one dataframe;
    gamma_ident_df can be a list of dataframes
    """

# Send data to AMPL
def ampl_send_data(ampl, gamma_df, gamma_ident_df, iv_df, identif):
    """
    Send data to AMPL
    Args:
        ampl (amplpy.AMPL): an AMPL object
        gamma_comb_df (pd.DataFrame): a dataframe of length k0 + k1
        iv_df (list): list of dataframes holding the identified estimands
    """
    # Send data to AMPL
    gamma_comb_df = gamma_df.join(gamma_ident_df)

    ampl.set_data(gamma_comb_df, "THETA")
    for i in range(len(identif)):
        ampl.set_data(iv_df[i], "IDENTIF_" + str(i))

# Write function to solve the model/run the ampl code
def ampl_eval(ampl_code, gamma_df, gamma_ident_df, iv_df, identif, quiet=False):
    ampl = AMPL()
    ampl.eval(ampl_code)

    # Send data
    ampl_send_data(ampl, gamma_df, gamma_ident_df, iv_df, identif)

    ampl.option["solver"] = "highs"
    ampl.option["solves_msg"] = 0


    # if quiet == True:
        # ampl.set_log_stream(None)
        # ampl.set_error_stream(None)
        # ampl.set_warning_stream(None)
        # ampl.set_results_stream(None)

    ampl.solve()
    assert ampl.get_value("solve_result") == "solved"

    beta_hi = ampl.get_objective('beta')
    print("Objective is:", beta_hi.value())
    argmax = ampl.get_variable('Theta_val').get_values().to_pandas()
    print("argmax:", argmax)

    return beta_hi.value(), argmax

# Generate relevant dataframes
def gen_identif_df(identif):
    return pd.DataFrame(
        [
            ("iv_1", identif)
        ],
        columns = ["IDENTIF", "iv"]
    ).set_index("IDENTIF")