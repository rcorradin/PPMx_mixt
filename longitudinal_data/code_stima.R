library(Rcpp)      
library(mcclust.ext)
sourceCpp("dist_MC.cpp")    
sourceCpp("CPPcode_long.cpp")

Y <- as.matrix(read.csv(file = "Y.csv"))
cens <- as.vector(read.csv(file = "cens.csv")[,1])
X_time <- as.matrix(read.csv(file = "X_time.csv"))
X_g_fun <- as.matrix(read.csv(file = "X_g_fun.csv"))
X_fix <- X_g_fun[,-1]
n_is <- as.vector(read.csv(file = "n_is.csv")[,1])

#--------------
# estimate the models

MC_for_lambda <- MC_routine(Y = as.matrix(X_g_fun), nrep = 10000, type_of_var = c(1,1,0,0,0,0,0,0), 
                            inv_cov_covs = solve(var(X_g_fun[,1:2])))
lambda_star <- 0.1 / mean(MC_for_lambda[,1])
est_model <- main_long(YY = log(Y), cens = rowSums(Y, na.rm = T) + 1, X_fixed = X_fix, X_random = X_time, 
                       X_g = X_g_fun, vartype = c(1,0,0,0,0,0,0), vartype_g = c(1,1,0,0,0,0,0,0), 
                       n_is = n_is, niter = 200, nburn = 100, thin = 1, 
                       k0k1 = diag(5,2), theta_0 = rep(0,2), a0 = 2, b0 = 1, 
                       kappa = 0.5, sigma = 0.2, s2v = 0.01, nu0 = 2, tau0 = 1,
                       g_fun = 3, lambda = lambda_star, power = 1, lambda0 = .2, 
                       n_importance = 100, napprox = 100, Sigma0 = diag(1, 7), nupd = 20, 
                       X_new_fixed = matrix(c(0,0,0,0)), X_new_random = matrix(c(0,0,0)), 
                       X_new_g = matrix(c(0,0,0,0)), inv_cov_covs = solve(var(X_g_fun[,1:2])),
                       n_is_new = c(0,0), prevision_full = F)
  
lLik <- as.matrix(est_model$LPML)
PSM <- psm(est_model$clust)

greedy_part <- minVI(psm = PSM, cls.draw=est_model$clust + 1, method = "draws")
est_part <- greedy_part$cl
table(est_part)  

