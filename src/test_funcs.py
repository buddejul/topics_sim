# Import all functons from funcs.py
from funcs import *
import pytest

def test_bernstein_basis():
    val = 10 * 0.5 ** 2 * (1-0.5)**3
    assert bern_bas(5, 2, 0.5) == val

def test_iv_slope():
    val = 0.074

    supp_z = np.array([0, 1, 2])
    f_z = np.array([0.5, 0.4, 0.1])
    prop_z = np.array([0.35, 0.6, 0.7])

    assert compute_estimand("iv_slope", m0_dgp, m1_dgp, 
                            supp_z = supp_z,
                            prop_z = prop_z,
                            f_z = f_z) == pytest.approx(val, 0.01)
    
def test_late():
    val = 0.046

    assert compute_estimand("late", m0_dgp, m1_dgp, u_lo=0.35, u_hi=0.9) == pytest.approx(val, 0.01)

def test_gamma_df():
    test_df = pd.DataFrame(
        [
            ("theta0_0",  0.000000),
            ("theta0_1", -0.454545),
            ("theta0_2", -0.545455),
            ("theta0_3",  0.000000),
            ("theta1_0",  0.000000),
            ("theta1_1",  0.454545),
            ("theta1_2",  0.545455),
            ("theta1_3",  0.000000),
        ],
        columns=["THETA", "gamma"],
    ).set_index("THETA")

    gamma_df = compute_gamma_df("late", "cs", u_lo = 0.35, u_hi = 0.90, u_part=np.array([0, 0.35, 0.60, 0.90, 1]))

    pd.testing.assert_frame_equal(gamma_df,test_df)