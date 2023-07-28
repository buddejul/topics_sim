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

# Gamma* linear maps
def gamma_star(
        md,
        d,
        estimand,
        u_lo=0,
        u_hi=1,
        supp_z=None,
        prop_z=None,
        f_z=None):
    
    """Gamma*0: the first term of the gamma* function for binary IV, no covariates
    """
    if estimand not in ["iv_slope", "late"]:
        raise ValueError("estimand must be either 'iv_slope' or 'late'")

    if estimand == "iv_slope":
        if supp_z is None or prop_z is None or f_z is None:
            raise ValueError("supp_z, prop_z, and f_z must be specified for iv_slope")

    if estimand == "late": 
        return integrate.quad(
            lambda u: md(u) * s_late(d, u, u_lo, u_hi), 0, 1)[0]

    elif estimand == "iv_slope":
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
    
    elif estimand == "ols_slope":
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
                     supp_z = 0, prop_z = 0, f_z = 0):

    if estimand not in ["iv_slope", "late"]:
        raise ValueError("estimand must be either 'iv_slope' or 'late'")
    
    if estimand == "late":
        a = gamma_star(m0, 0, estimand, u_lo = u_lo, u_hi = u_hi)
        b = gamma_star(m1, 1, estimand, u_lo = u_lo, u_hi = u_hi)

    elif estimand == "iv_slope":
        a = gamma_star(m0, 0, estimand, 
        supp_z = supp_z, prop_z = prop_z, f_z = f_z)
        
        b = gamma_star(m1, 1, estimand, 
        supp_z = supp_z, prop_z = prop_z, f_z = f_z) 

    return a + b

# Function writing AMPL model code

# Function sending data to AMPL

def compute_gamma_df(target, basis, k0 = None, k1 = None, u_part = None, u_lo=None, u_hi=None,
                     supp_z = None, prop_z = None, f_z = None):
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

    Returns:
        gamma_df (pd.DataFrame): a dataframe of length k0 + k1
    """
    
    if target not in ["iv_slope", "late"]:
        raise ValueError("target must be either 'iv_slope' or 'late'")
    
    if basis not in ["bernstien", "cs"]:
        raise ValueError("basis must be either 'bernstein' or 'cs'")
    
    if target == "late" and (u_lo is None or u_hi is None):
        raise ValueError("u_lo and u_hi must be specified for late target")
    
    if basis == "cs" and u_part is None:
        raise ValueError("u_part must be specified for cs basis")
    
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
                        u_lo = u_lo, u_hi = u_hi)
                    
                    gamma1[i] = gamma_star(func, 1, target, 
                        u_lo = u_lo, u_hi = u_hi)

                elif target == "iv_slope":
                    gamma0[i] = gamma_star(func, 0, target, 
                        supp_z = supp_z, prop_z = prop_z, f_z = f_z)
                    
                    gamma1[i] = gamma_star(func, 1, target, 
                        supp_z = supp_z, prop_z = prop_z, f_z = f_z)
    
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
def ampl_code(min_or_max):
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
    ampl_code += "set IDENTIF;\n"

    # Define parameters
    ampl_code += "param gamma {THETA};\n"
    ampl_code += "param gamma_ident {THETA};\n"
    ampl_code += "param iv {IDENTIF};\n"

    # Choice variable
    ampl_code += "var Theta_val {j in THETA} >= 0, <= 1;\n"

    # Objective function    
    ampl_code += min_or_max + " beta: sum {j in THETA} gamma[j] * Theta_val[j];\n"

    # Constraints
    # TODO write code for set with more than 1 identified parameter
    ampl_code += "subject to Identified {i in IDENTIF}:\n"
    ampl_code += "sum {j in THETA} gamma_ident[j] * Theta_val[j] == iv[i];\n"

    return ampl_code

# Send data to AMPL
def ampl_send_data(ampl, gamma_df, gamma_ident_df, iv_df):
    """
    Send data to AMPL
    Args:
        ampl (amplpy.AMPL): an AMPL object
        gamma_comb_df (pd.DataFrame): a dataframe of length k0 + k1
        iv_df (pd.DataFrame): a dataframe of length k0 + k1
    """
    # Send data to AMPL
    gamma_comb_df = gamma_df.join(gamma_ident_df)

    ampl.set_data(gamma_comb_df, "THETA")
    ampl.set_data(iv_df, "IDENTIF")

# Write function to solve the model/run the ampl code
def ampl_eval(ampl_code, gamma_df, gamma_ident_df, iv_df, quiet=False):
    ampl = AMPL()
    ampl.eval(ampl_code)

    # Send data
    ampl_send_data(ampl, gamma_df, gamma_ident_df, iv_df)

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