from funcs import *

gamma_df = compute_gamma_df("late", "cs", u_lo = 0.35, u_hi = 0.90, u_part=np.array([0, 0.35, 0.60, 0.90, 1]))
print(gamma_df)

# Print column types of gamma_df
print(gamma_df.dtypes)
