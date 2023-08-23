from plot_funcs import *
from funcs import m0_dgp, m1_dgp
import cProfile
import timeit

# Define DGP and other parameters
# Target
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

stmt = '''iv_part_id(target = target,identif = ["cross","cross","cross","cross","cross","cross"], shape = ["decr"],basis = basis,m0_dgp=m0_dgp,m1_dgp=m1_dgp,u_lo_target = u_lo_target,u_hi_target = u_hi_target,u_part = u_part,supp_z = supp_z,f_z=f_z,dz_cross = dz_cross,prop_z=prop_z,)'''
stmt2 = '''iv_part_id(target = target,identif = ["cross","cross","cross","cross","cross","cross"], shape = ["decr"],basis = basis,m0_dgp=m0_dgp,m1_dgp=m1_dgp,u_lo_target = u_lo_target,u_hi_target = u_hi_target,u_part = u_part,supp_z = supp_z,f_z=f_z,dz_cross = dz_cross,prop_z=prop_z,analyt_int=True)'''

cProfile.run(stmt2, sort='cumtime')

setup = '''
gc.enable()
from funcs import m0_dgp, m1_dgp
from iv_part_id import iv_part_id
import numpy as np
target = "late"
u_lo_target = 0.35
u_hi_target = 0.9
basis = "cs"
supp_z=[np.array([0, 1, 2])]
f_z=[np.array([0.5, 0.4, 0.1])]
prop_z=[np.array([0.35, 0.6, 0.7])]
u_part=np.array([0, 0.35, 0.60, 0.7, 0.90, 1])
dz_cross = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
'''

t = timeit.Timer(stmt, setup)

print(t.repeat(repeat=3, number=100))