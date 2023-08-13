# Test all the bounds derived in the paper
from funcs import *
from iv_part_id import *
import pytest

# Bounds
bounds_fig2 = (-0.421, 0.500)
bounds_fig3 = (-0.411, 0.500)
bounds_fig4 = (-0.320, 0.407)
bounds_fig5 = (-0.138, 0.407)
bounds_fig6 = (-0.095, 0.077)

# Setup

target = "late"
u_lo_target = 0.35
u_hi_target = 0.9

basis = "cs"

# Instrument
supp_z=[np.array([0, 1, 2])]
f_z=[np.array([0.5, 0.4, 0.1])]
prop_z=[np.array([0.35, 0.6, 0.7])]

# Partition of u for constant spline approximation
u_part=np.array([0, 0.35, 0.60, 0.7, 0.90, 1])

# Cross-moments
dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

def test_bounds_fig2():
    results = iv_part_id(
        target = target,
        identif = ["iv_slope"],
        basis = basis,
        m0_dgp=m0_dgp,
        m1_dgp=m1_dgp,
        u_lo_target = 0.35,
        u_hi_target = 0.9,
        u_part = u_part,
        supp_z = supp_z,
        f_z=f_z,
        prop_z=prop_z,
        analyt_int=True
    )

    beta_lo, beta_hi = results[0][0], results[1][0]

    assert (beta_lo, beta_hi) == pytest.approx(bounds_fig2, 0.001)

supp_z=[np.array([0, 1, 2]),np.array([0, 1, 2])]
f_z=[np.array([0.5, 0.4, 0.1]),np.array([0.5, 0.4, 0.1])]
prop_z=[np.array([0.35, 0.6, 0.7]),np.array([0.35, 0.6, 0.7])]

def test_bounds_fig3():
    results = iv_part_id(
        target = target,
        identif = ["iv_slope", "ols_slope"],
        basis = basis,
        m0_dgp=m0_dgp,
        m1_dgp=m1_dgp,
        u_lo_target = 0.35,
        u_hi_target = 0.9,
        u_part = u_part,
        supp_z = supp_z,
        f_z=f_z,
        prop_z=prop_z,
        analyt_int=True
    )

    beta_lo, beta_hi = results[0][0], results[1][0]

    assert (beta_lo, beta_hi) == pytest.approx(bounds_fig3, 0.001)

# Figure 5
u_part=np.array([0, 0.35, 0.60, 0.7, 0.90, 1])
dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

def test_bounds_fig5():
    results = iv_part_id(
        target = target,
        identif = ["cross","cross","cross","cross","cross","cross"], # the set of identified estimands
        basis = basis,
        m0_dgp=m0_dgp,
        m1_dgp=m1_dgp,
        u_lo_target = 0.35,
        u_hi_target = 0.9,
        u_part = u_part,
        supp_z = supp_z,
        f_z=f_z,
        prop_z=prop_z,
        dz_cross=dz_cross,
        analyt_int=True
    )

    beta_lo, beta_hi = results[0][0], results[1][0]

    assert (beta_lo, beta_hi) == pytest.approx(bounds_fig5, 0.01)

# Figure 6
def test_bounds_fig6():
    results = iv_part_id(
        target = target,
        identif = ["cross","cross","cross","cross","cross","cross"], # the set of identified estimands
        basis = basis,
        shape = ["decr"],
        m0_dgp=m0_dgp,
        m1_dgp=m1_dgp,
        u_lo_target = 0.35,
        u_hi_target = 0.9,
        u_part = u_part,
        supp_z = supp_z,
        f_z=f_z,
        prop_z=prop_z,
        dz_cross=dz_cross,
        analyt_int=True
    )

    beta_lo, beta_hi = results[0][0], results[1][0]

    assert (beta_lo, beta_hi) == pytest.approx(bounds_fig6, 0.01)
