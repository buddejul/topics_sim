import pandas as pd
import numpy as np

def table_by_tol(inpath, outpath, N, R, tolerances, true_beta):

    # Load numpy arrays for
    tolerances = [N**(-2), N**(-1), N**(-1/2), N**(-1/4)]

    # Load the results, calculate mean and SD
    mean = np.zeros((len(tolerances), 2))
    sd = np.zeros((len(tolerances), 2))

    for i, tol in enumerate(tolerances):
        beta_lo = np.load(inpath + "/beta_lo_n" + str(N) +"_r" + str(R) + "_tol" + str(tol) + ".npy")
        beta_hi = np.load(inpath + "/beta_hi_n" + str(N) +"_r" + str(R) + "_tol" + str(tol) + ".npy")
        mean[i, 0] = np.mean(beta_lo)
        mean[i, 1] = np.mean(beta_hi)
        sd[i, 0] = np.std(beta_lo)
        sd[i, 1] = np.std(beta_hi) 

    # Put into pandas dataframe together with tolerances
    df = pd.DataFrame(mean, columns = ["beta_lo", "beta_hi"])
    df["sd_lo"] = sd[:, 0]
    df["sd_hi"] = sd[:, 1]
    df["tolerance"] = tolerances

    # Compute MSE for beta_lo and beta_hi (bias^2 + variance)
    beta_lo = true_beta[0]
    beta_hi = true_beta[1]
    df["mse_lo"] = (df["beta_lo"] - beta_lo)**2 + df["sd_lo"]**2
    df["mse_hi"] = (df["beta_hi"] - beta_hi)**2 + df["sd_hi"]**2

    # Output df to latex table in inpath folder
    # df.style.to_latex(inpath + "/table_by_tol.tex")

    # Output df to latex table in inpath folder and add true beta as note
    # Rename the columns
    df.columns = ["$\hat{\\beta}_{lo}$", "$\hat{\\beta}_{hi}$", "$sd_{lo}$", "$sd_{hi}$", "tolerance", "MSE$_{lo}$", "MSE$_{hi}$"]
    
    # Same as above but add (N = N, R = R) to caption
    df.style.set_caption("True Lower Bound: " + str(np.round(beta_lo,3)) + " True Upper Bound: " + str(np.round(beta_hi,3)) + " (N = " + str(N) + ", R = " + str(R) + ")").to_latex(outpath + "/table_by_tol_N" + str(N) + "_R" + str(R) +".tex")    

    

    return None
