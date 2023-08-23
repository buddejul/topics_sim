# Functions required for estimation of the identifid set
from funcs import *
import numpy as np
import pandas as pd

def estimate_iv_slope_nocons(y, z, d):
    """Estimate the IV slope parameter beta for binary endogenous variable and 
    scalar instrument z; no constant in the model.

    Args:
        y (np.array): outcome variable
        z (np.array): instrument
        d (np.array): endogenous variable

    Returns:
        float: IV slope parameter beta
    """
    cov_yz = np.cov(y, z, bias=True)[0, 1]
    cov_dz = np.cov(d, z, bias=True)[0, 1]

    return cov_yz / cov_dz

def estimate_ols_slope_nocons(y, d):
    """Estimate the OLS slope parameter beta for binary endogenous variable; 
    no constant in the model.

    Args:
        y (np.array): outcome variable
        d (np.array): endogenous variable

    Returns:
        float: OLS slope parameter beta
    """
    cov_yd = np.cov(y, d, bias=True)[0, 1]
    var_d = np.var(d, ddof=0)

    return cov_yd / var_d

def estimate_cross(y, z, d, d_val, z_val):
    a = d == d_val
    b = z == z_val

    return np.mean(y * a * b)

def estimate_beta(y, z, d, s):
    """ Estimate beta based on the sample analogue of beta_s = E[s(D,Z)Y]
    
    Args:
        y (np.array): outcome variable
        z (np.array): instrument
        d (np.array): endogenous variable
        s (function): function of d and z

    Returns:
        float: beta
    """

    return np.mean(s(d, z) * y)

# Estimatate weights
def estimate_prop_z(z, d):
    """ Estimate propensity score as a function of z"""
    supp_z = np.unique(z)
    supp_z.sort() # make sure the propensity is ordered by the support of z

    p = np.zeros(len(supp_z))
    for i, z_val in enumerate(supp_z):
        p[i] = np.mean(d[z == z_val])

    return p

def estimate_f_z(z):
    """ Estimate probability mass function of z"""
    supp_z = np.unique(z)
    supp_z.sort() # make sure f_z is ordered by the support of z

    f_z = np.zeros(len(supp_z))
    for i, z_val in enumerate(supp_z):
        f_z[i] = np.mean(z == z_val)

    return f_z

def estimate_moments(y, z, d):
    ez = np.mean(z)

    ed = np.mean(d)
    var_d = np.var(d)

    edz = np.mean(d*z)
    cov_dz = edz - ed * ez

    return ez, ed, var_d, cov_dz

def estimate_w_ds(z, d, p, s, d_val): 
    """
    Returns estimate of w_0s (a function of u) as defined in the Appendix of MST (2019)

    Args:
        z (np.array): data on the instrument
        d (np.array): data on the treatment
        p (np.array): Estimated propensity scores of length len(supp_z)
        s (string): string indicating the IV-like specification
        d_val (integer): the value of d (0 or 1)

    Returns:
        function: estimate of w_0s
    """

    if len(p) != len(np.unique(z)):
        raise ValueError("p must be of length len(supp_z)")
    
    if s not in ["ols_slope", "iv_slope", "cross"]:
        raise ValueError("s must be one of 'ols_slope', 'iv_slope', 'cross'")

    supp_z = np.unique(z)
    supp_z.sort()

    if s == "ols_slope": 
        var_d = np.var(d)
        ed = np.mean(d)

        def s(d,z): return s_ols_slope(d,ed=ed,var_d=var_d)

    if s == "iv_slope":
        ez = np.mean(z)
        ed = np.mean(d)
        edz = np.mean(d*z)
        cov_dz = edz - ed * ez

        def s(d,z): return s_iv_slope(z, ez=ez, cov_dz=cov_dz)

    if s == "cross":
        dz_cross = (0,1)

        def s(d,z): return s_cross(d, z, dz_cross=dz_cross)

    
    if d_val == 0:
        def w_0s(u,z): return s(d_val,z) * (u > p[np.where(supp_z == z)[0][0]])
        return w_0s

    if d_val == 1:
        def w_1s(u,z): return s(d_val,z) * (u <= p[np.where(supp_z == z)[0][0]])
        return w_1s

# Estimate gamma_ds
def estimate_gamma_ds(d, z, basis, d_val, s, u_part=None, u_lo=None, u_hi=None, dz_cross=None):

    if basis == "cs" and (u_lo is None or u_hi is None):
        raise ValueError("u_lo and u_hi must be specified for constant splines")

    if s == "cross" and dz_cross is None:
        raise ValueError("dz_cross must be specified for cross terms")

    if d_val not in [0, 1]:
        raise ValueError("d_val must be 0 or 1")
    
    if basis == "cs" and u_part is None:
        raise ValueError("u_part must be specified for constant splines")

    # Estimate propensity scores
    p = estimate_prop_z(z, d)

    # Get vector of p corresponding to z
    supp_z = np.unique(z)
    supp_z.sort() # make sure the propensity is ordered by the support of z
    z_p = p[np.searchsorted(supp_z, z)]

    # For constant splines don't need to estimate the weights w_ds, only need
    # the estimated IV-like specifications
    if basis == "cs":

        if s == "ols_slope": 
            var_d = np.var(d)
            ed = np.mean(d)

            def s(d,z): return s_ols_slope(d,ed=ed,var_d=var_d)

        if s == "iv_slope":
            ez = np.mean(z)
            ed = np.mean(d)
            edz = np.mean(d*z)
            cov_dz = edz - ed * ez

            def s(d,z): return s_iv_slope(z, ez=ez, cov_dz=cov_dz)

        if s == "cross":
            def s(d,z): return s_cross(d, z, dz_cross=dz_cross)
 
        # TODO potential bug: u_hi - u_lo for both cases?
        if d_val == 0:
            return np.mean(
                (u_lo > z_p) * s(d_val, z) * (u_hi - u_lo)
            )

        if d_val == 1:
            return np.mean(
                (u_hi <= z_p) * s(d_val, z) * (u_hi - u_lo)
            )


    # TODO implement for Bernstein polynomials

def compute_gamma_df_estimation(y, z, d, target, basis, k0 = None, k1 = None, u_lo=None, u_hi=None,
                     dz_cross = None, u_part = None):
    """
    Compute gamma* evaluated at different basis functions
    Args:
        y (np.array): outcome variable
        z (np.array): instrument
        d (np.array): endogenous variable
        target (str): the target estimand
        k0 (int): degree of the polynomial for the d = 0 MTE
        k1 (int): degree of the polynomial for the d = 1 MTE
        basis (str): the basis function to use
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
    
    if target == "cross" and dz_cross is None:
        raise ValueError("dz_cross must be specified for cross-moment")
    
    if basis == "cs" and u_part is None:
        raise ValueError("u_part must be specified for constant splines")

    # Estimate propensity scores
    p = estimate_prop_z(z, d)

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

                # TODO implement late (note this is target however)

                if target == "iv_slope" or target == "ols_slope":
                    gamma0[i] = estimate_gamma_ds(d, z, "cs", 0, target,
                                    u_part = u_part, u_lo = u_part[i], u_hi = u_part[i+1])
                    
                    gamma1[i] = estimate_gamma_ds(d, z, "cs", 1, target,
                                    u_part = u_part, u_lo = u_part[i], u_hi = u_part[i+1])
                    
                elif target == "cross":
                    gamma0[i] = estimate_gamma_ds(d, z, "cs", 0, target,
                                    u_part = u_part, u_lo = u_part[i], u_hi = u_part[i+1], 
                                    dz_cross=dz_cross)
                    
                    gamma1[i] = estimate_gamma_ds(d, z, "cs", 1, target,
                                    u_part = u_part, u_lo = u_part[i], u_hi = u_part[i+1], 
                                    dz_cross=dz_cross)
    

    # Generate column vector of names for d=0, d=1, and k1, k2
    if basis == "bernstein":
        d0_gamma = ["theta0_" + str(i) for i in range(k0)]
        d1_gamma = ["theta1_" + str(i) for i in range(k1)]
        d_gamma = d0_gamma + d1_gamma

    if basis == "cs":
        d0_gamma = ["theta0_" + str(i) for i in range(len(u_part) - 1)]
        d1_gamma = ["theta1_" + str(i) for i in range(len(u_part) - 1)]
        d_gamma = d0_gamma + d1_gamma

    # Concatenate d, theta0 + theta1
    gamma = np.concatenate((gamma0, gamma1))

    df = pd.DataFrame(gamma, d_gamma, columns=['gamma'])
    df.index.rename("THETA", inplace=True)

    return df

# Write the function generating the first AMPL LP for the RHS of the constraint
def ampl_code_rhs_constraint(identif):
    """
    Generate AMPL code for the probem in the RHS of the constraint in estimation
    see MST equation (27) and following discussion.

    For this we only need the dataframes

    Note the target is in absolute value; to code this we are using tricks from
    from the following sources (adding additional slack variables).

    See the first answer in:
    https://math.stackexchange.com/questions/432003/converting-absolute-value-program-into-linear-program

    Also see here:
    https://groups.google.com/g/ampl/c/FKqSLiPyHP0

    And here (most helpful for understanding the "trick"):
    https://lpsolve.sourceforge.net/5.1/absolute.htm
    

    Approach: The objective in the case of S identified estimands is given by
        min abs(sum([x * g_1]) - b_1) + abs(sum([x * g_2]) - b_2) + ... + abs(sum([x * g_S]) - b_S) 

    The tricky bit is the absolute value. Rewrite the objective as follows:
        min abs(X_1) + abs(X_2) + ... + abs(X_S)

    Now we add S variables called X_s' and for each variable two constraints:
        min X_1' + X_2' + ... + X_S'
        s.t.
        X_s' >= X_s
        X_s' >= -X_s

        for all s = 1, ..., S

    Note X' >= 0 else the constraints cannot be satisfied.

    Now we can think about three cases:
        - X_s>0: Then -X_s<0 hence constraint two holds (X's are non-negative).
            Further, X_s <= X_s' only constraints X_s' to be at least as large as
            X_s. However, the objective minimizes over X_s', so X_s' = X.
        - X_s<0: Then constraint is automatically satisfied. Constraint two implies
            that X_s' must be as large as -X_s>0, but the objective minimizes over X_s',
            hence X_s' = -X_s.
        - X_s=0: The constraints are always satisfied and the minimal X_s' is 0.
    Hence, X_s' = abs(X_s).

    Now plug in the original definition of X_s to get the desired program.
    Note we essentially moved all the "real" variables to the constraints!

    Args:
        identif (list): list of identified estimands
    
    Returns:
        ampl_code (str): AMPL code for problem in the RHS of the constraint
    """
    
    # Generate AMPL code for the identified set
    # Define sets
    ampl_code = "reset;\n"
    ampl_code += "set THETA;\n"
    for i in range(len(identif)):
        ampl_code += "set IDENTIF_" + str(i) + ";\n"

    # Define parameters
    for i in range(len(identif)):
        ampl_code += "param gamma_ident_" + str(i) + " {THETA};\n"
        ampl_code += "param val_identif_" + str(i) + " {IDENTIF_" + str(i) + "};\n"

    # Choice variable (note this is equivalent to Y in [0,1] constraint, see MST 2018 Appendix)
    ampl_code += "var Theta_val {j in THETA} >= 0, <= 1;\n"

    # Add a "dummy" variable for each identified estimand (trick to get abs value)
    for i in range(len(identif)):
        ampl_code += "var X_" + str(i) + " >= 0;\n"

    # Objective function    
    ampl_code += "minimize beta: "
    for i in range(len(identif)):
        ampl_code += "X_" + str(i)
        if i < len(identif) - 1:
            ampl_code += " + "

    ampl_code += ";\n"
    
    # Constraints to get absolute value into objective function
    for i in range(len(identif)):
        # Foreach identified estimand:
            # The LHS computes: - estimated estimand + implied estimand by minimizer (or "-" that for the second one)
            # The RHS is the slack variable X_i

        ampl_code += "subject to abs_1_" + str(i) + " {i in IDENTIF_" + str(i) + "}:\n"
        ampl_code += "(-val_identif_" + str(i) + "[i] + sum {j in THETA} gamma_ident_" + str(i) + "[j] * Theta_val[j]) <= X_" + str(i) + ";\n"

        ampl_code += "subject to abs_2_" + str(i) + " {i in IDENTIF_" + str(i) + "}:\n"
        ampl_code += "-(-val_identif_" + str(i) + "[i] + sum {j in THETA} gamma_ident_" + str(i) + "[j] * Theta_val[j]) <= X_" + str(i) + ";\n"

    return ampl_code

def ampl_code_estimation(min_or_max, identif, inf, tol):
    """
    Generate AMPL code for the estimation problem
    see MST equation (27) and following discussion.

    Now the constraint is in absolute value; to code this we are using tricks from
    from the following sources (adding additional slack variables).

    And here (most helpful for understanding the "trick"):
    https://lpsolve.sourceforge.net/5.1/absolute.htm
    
    Approach: The objective given the target is given by
        min (max) sum([x * gt_1])    # min (max) estimate of beta consistent with constraints 

    Now the constraint features absolute values:
        abs(sum([x * g_1]) - b_1) + abs(sum([x * g_2]) - b_2) + ... + abs(sum([x * g_S]) - b_S) <= d_min + kappa

        or

        X1 + X2 + ... + XS <= d_min + kappa

        where the LHS are deviations from identified estimates and the RHS is 
        the minimum distance (d_min) and a tolerance (kappa).

        To get an absolute value representation replace by a dummy variable:
        
        (1) X_1' + X_2' + ... + X_S' <= d_min + kappa

        But then add a constraint that ensures X_1' = abs(X_1) and so on:
            (2) X_s' >= X_s
            (3) X_s' >= -X_s

        for all s = 1, ..., S.

    Args:
        identif (list): list of identified estimands
        inf (float): the minimum distance
        kappa (float): the tolerance
    
    Returns:
        ampl_code (str): AMPL code for problem in the RHS of the constraint
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

    # Add a "dummy" variable for each identified estimand (trick to get abs value)
    for i in range(len(identif)):
        ampl_code += "var X_" + str(i) + " >= 0;\n"

    # Objective function (same as for identification; only that gamma are estimated)
    ampl_code += min_or_max + " beta: sum {j in THETA} gamma[j] * Theta_val[j];\n"

    # First constraint sums up all the X_s' (absolute values) and requires small than
    # infimum + tolerance
    ampl_code += "subject to distance: "
    for i in range(len(identif)):
        ampl_code += "X_" + str(i)
        if i < len(identif) - 1:
            ampl_code += " + "

    ampl_code += " <= " + str(inf) + " + " + str(tol) +"\n"
    ampl_code += ";\n"

    # Constraints to get absolute value into objective function
    for i in range(len(identif)):

        # Foreach identified estimand:
            # The LHS computes: - estimated estimand + implied estimand by minimizer (or "-" that for the second one)
            # The RHS is the slack variable X_i

        ampl_code += "subject to abs_1_" + str(i) + " {i in IDENTIF_" + str(i) + "}:\n"
        ampl_code += "(-val_identif_" + str(i) + "[i] + sum {j in THETA} gamma_ident_" + str(i) + "[j] * Theta_val[j]) <= X_" + str(i) + ";\n"

        ampl_code += "subject to abs_2_" + str(i) + " {i in IDENTIF_" + str(i) + "}:\n"
        ampl_code += "-(-val_identif_" + str(i) + "[i] + sum {j in THETA} gamma_ident_" + str(i) + "[j] * Theta_val[j]) <= X_" + str(i) + ";\n"

    return ampl_code

