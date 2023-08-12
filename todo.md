# Todos

## Bugs/Improvements

- Fix the indexing for OLS and IV arrays --> have to always specify IV before ols
- Fix IV Slope coefficient for binary based on discrete

## Identification

General:

- Get multiple identified estimands to work
- Write function that plots figures
- Option to plot maximizing MTRs
- Closed form Bernstein polynomials
- Implement binarized instrument via baseline IV and binarized IV

Specific to figures:

- Figure 2: Plot IV Slope Weights
- Figure 3: Implement OLS Slope Weights
- Figure 4: IV Slope with discrete Instruments (should already work)
- Figure 5: Weights for DZ cross-moments
- Figure 6: Shape restriction -- decreasing
- Figure 7: Polynomial MTRs

## Estimation/Identification based on data

## Inference: Bootstrap