# Simulation results using the official IVMTE R package by the authors
# For details see https://github.com/jkcshea/ivmte

# Some required packages
# install.packages("ivmte") # main package
# install.packages("slam", repos = "https://cloud.r-project.org") # for splines
# gurobi R api path, for details see github package linked above
# install.packages("C:/gurobi1002/win64/R/gurobi_10.0-2.zip", repos = NULL)
# install.packages("splines2") # splines
# install.packages("lsei") # least squares for optimization
# install.packages("knitr") # table display
# install.packages("AER") # for ivreg
# install.packages("data.table") # used in analysis

library(ggplot2)
library(AER)
library(gurobi)
library(ivmte)
library(data.table)

# Bernstein basis
bern_bas <- function(n, v, x){
  choose(n, v) * x^v * (1 - x)^(n-v)
}

# DGP from MST ECMA 2018
# MTR for d = 0
m0_dgp <- function(u){
  0.6 * bern_bas(2, 0, u) + 0.4 * bern_bas(2, 1, u) + 0.3 * bern_bas(2, 2, u)
}

# MTR for d = 1
m1_dgp <- function(u){
  0.75 * bern_bas(2, 0, u) + 0.5 * bern_bas(2, 1, u) + 0.25 * bern_bas(2, 2, u)
}

# Function to simulate data from DGP
sim_data <- function(n, supp, f, p, m0, m1){
  
  u = runif(n, 0, 1)
  z = sample(supp, n, replace=TRUE, prob=f)
  pz = p[z+1]
  d = pz >= u
  
  y0 = m0(u)
  y1 = m1(u)
  
  # Note: Instead of simulating binary Y I directly simulate y to be equal to the
  # MTR for a given u (i.e. average Y for d,u)

  y = rep(0, n)
  y[d==0] = y0[d==0]
  y[d==1] = y1[d==1]
  
  return(list(y=y, d=d, z=z))
  
}

# Parameters of the DGP from MST ECMA 2018
supp = c(0, 1, 2)
f = c(0.5, 0.4, 0.1)
p = c(0.35, 0.6, 0.7)

# Function for simulation exercise (grid sizes are ivmte internal parameters)
sim <- function(n, reps, supp, f, p, m0_dgp, m1_dgp, 
                init_grid=1000, audit_grid=2000, u_hi){
  
  bounds = matrix(nrow=reps, ncol=2)
  
  for (i in seq(1, reps)){
    print(i)
    data = sim_data(n, supp, f, p, m0_dgp, m1_dgp)
    data = as.data.frame(do.call(cbind, data))
    data = data.table(data)
    data
    
    # Create new columns for d==0 and d==1
    data[, d0 := ifelse(d == 0, 1, 0)]
    data[, d1 := ifelse(d == 1, 1, 0)]
    
    # Create new columns for z==0, z==1, and z==2
    data[, z0 := ifelse(z == 0, 1, 0)]
    data[, z1 := ifelse(z == 1, 1, 0)]
    data[, z2 := ifelse(z == 2, 1, 0)]
    
    # estimate propensity score
    pmod = lm(d~0+z0+z1+z2, data=data)
    p_est <- coef(pmod)
    p_est
    
    u_part = sort(c(p_est, 0.35, u_hi))
    
    # The saturated specificaiton (or list of specifications) should correspond
    # to the sharp bounds using all the E[y(D==d)(Z==z)] cross-moments
    args <- list(data = data,
                 ivlike =  c(
                   y~0+d0:z0,
                   y~0+d0:z1,
                   y~0+d0:z2,
                   y~0+d1:z0,
                   y~0+d1:z1,
                   y~0+d1:z2
                 ),
                 target = "genlate",
                 genlate.lb = 0.35,
                 genlate.ub = u_hi,
                 m0 = ~ uSplines(degree = 0, knots = u_part),
                 m1 = ~ uSplines(degree = 0, knots = u_part),
                 propensity = d ~ 0 + z0 + z1 + z2,
                 initgrid.nx=init_grid, initgrid.nu=init_grid,
                 audit.nx=audit_grid, audit.nu=audit_grid,
                 m0.lb=0, 
                 m0.ub=1, 
                 m1.lb=0, 
                 m1.ub=1
                 ,criterion.tol=1/n
                 # ,point=FALSE
                 
                 # ,soft=TRUE,
                 )
    
    r <- do.call(ivmte, args)
    bounds[i,] =  r$bounds
  }
  
  return(bounds)
}

# Run the simulation
n = 1000
reps = 50
results = sim(n, reps, supp, f, p, m0_dgp, m1_dgp, 
              init_grid=10000, audit_grid=10000,
              u_hi=0.9)

# Save the results in data.table
df_res <- data.table(results)

# Plot a histogram of the lower bound
ggplot(df_res, aes(x=V1)) + geom_histogram(fill="blue")

# Plot a histogram of the upper bound
ggplot(df_res, aes(x=V2)) + geom_histogram(fill="red")

mean(df_res$V1)
mean(df_res$V2)

#################### 
# Compute IV slope to check data simulation
data <- sim_data(10000, supp, f, p, m0_dgp, m1_dgp)

sim_iv_slope <- function(n, reps, supp, f, p, m0_dgp, m1_dgp){

  iv_sim = vector(length=reps)
  iv_like = matrix(nrow=reps, ncol=6)
  p_z = matrix(nrow=reps, ncol=3)
  
  for (i in seq(1, reps)){
    data = sim_data(n, supp, f, p, m0_dgp, m1_dgp)
    iv_sim[i] = cov(data$y, data$z) / cov(data$d, data$z)
    
    
    data = as.data.frame(do.call(cbind, data))
    data = data.table(data)
    # Create new columns for d==0 and d==1
    data[, d0 := ifelse(d == 0, 1, 0)]
    data[, d1 := ifelse(d == 1, 1, 0)]
    
    # Create new columns for z==0, z==1, and z==2
    data[, z0 := ifelse(z == 0, 1, 0)]
    data[, z1 := ifelse(z == 1, 1, 0)]
    data[, z2 := ifelse(z == 2, 1, 0)]
    
    ivmod = lm(y ~ 0+d0:z0 + d0:z1 + d0:z2 + d1:z0 + d1:z1 + d1:z2, 
               data=data)
    
    iv_like[i,] <- coef(ivmod)
      
    pmod = lm(d~0+z0+z1+z2, data=data)
    p_z[i,] <- coef(pmod)
  }

  return(list(iv_sim=iv_sim, p_z = p_z, iv_like = iv_like))
}

iv_sim = sim_iv_slope(n=1000, reps=1000, supp, f, p, m0_dgp, m1_dgp)

mean(iv_sim$iv_sim)
colMeans(iv_sim$p_z)
colMeans(iv_sim$iv_like)

# True IV slope is 0.074

#########################
# Some data checking
data <- sim_data(10000, supp, f, p, m0_dgp, m1_dgp)
data = as.data.frame(do.call(cbind, data))
data = data.table(data)
# Create new columns for d==0 and d==1
data[, d0 := ifelse(d == 0, 1, 0)]
data[, d1 := ifelse(d == 1, 1, 0)]

# Create new columns for z==0, z==1, and z==2
data[, z0 := ifelse(z == 0, 1, 0)]
data[, z1 := ifelse(z == 1, 1, 0)]
data[, z2 := ifelse(z == 2, 1, 0)]

ivmod = lm(y ~ 0+d0:z0 + d0:z1 + d0:z2 + d1:z0 + d1:z1 + d1:z2, 
           data=data)
coef(ivmod)
ivreg(data = data, y ~ d | z )

lm(y ~ 0 + d0:z0, data=data)
