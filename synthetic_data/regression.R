library(Rcpp)

Rcpp::sourceCpp('dist_MC.cpp')
Rcpp::sourceCpp('CPPcode.cpp')
Rcpp::sourceCpp('utilities.cpp')

library(mvtnorm)
library(ggplot2)
library(mcclust.ext)

#------------ robusteness REGRESSION --------
#--------------------------------------------

# generate the data
#---------------------

set.seed(123)
group_1 <- cbind(rmvnorm(75, c(-3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 75, prob = c(0.9, 0.1), replace = T),
                 sample(c(0,1), size = 75, prob = c(0.9, 0.1), replace = T))
group_1 <- cbind(apply(group_1, 1, function(x) rnorm(1, 1 + x %*% c(5, 2, 1, 0), .5)), group_1)

group_2<- cbind(rmvnorm(75, c(0,0), diag(.5, 2)), 
                 sample(c(0,1), size = 75, prob = c(0.5, 0.5), replace = T),
                 sample(c(0,1), size = 75, prob = c(0.5, 0.5), replace = T))
group_2 <- cbind(apply(group_2, 1, function(x) rnorm(1, 4 + x %*% c(2, -2, 1, -1), .5)), group_2)

group_3 <- cbind(rmvnorm(50, c(3,3), diag(.5, 2)), 
                 sample(c(0,1), size = 50, prob = c(0.1, 0.9), replace = T),
                 sample(c(0,1), size = 50, prob = c(0.1, 0.9), replace = T))
group_3 <- cbind(apply(group_3, 1, function(x) rnorm(1,  -1 + x %*% c(-5, -2, -1, 1), .5)), group_3)

data_rob <- rbind(group_1, group_2, group_3)

Xnew <- rbind(c(1,1,0,0), cbind(rmvnorm(20, c(-3,3), diag(1.5, 2)), 
              sample(c(0,1), size = 10, prob = c(0.9, 0.1), replace = T),
              sample(c(0,1), size = 10, prob = c(0.9, 0.1), replace = T)))
grid <- seq(-15, 15, by = 0.1)

Ystar <- apply(apply(Xnew, 1, function(y) sapply(grid, function(x) 
  dnorm(x, 1 + y %*% c(5, 2, 1, 0), 0.5)  * 1+ 
    dnorm(x, 4 + y %*% c(2, -2, 1, -1), 0.5) * 0 + 
    dnorm(x, -1 + y %*% c(-3, 0, -1, 1), 0.5) * 0)), 2, function(z) z %*% grid / sum(z))
true_Eval <- as.numeric(c(1,1,1,0,0) %*% c(4,2, -2, 1, -1))

# choosing lambda
#---------------------

lambda_temp <- MC_routine(Y = data_rob[,-1], nrep = 10000, type_of_var = c(1,1,0,0))
lambda_opt <- .5 / mean(lambda_temp[,1])

# model specification
#---------------------

niter <- 15000
nburn <- 10000
thin <- 1
  
kappa <- 0.3
sigma <- 0.2
B0 <- diag(100, 5)
m0 <- rep(0, 5)
a0 <- 2
b0 <- 1

# run the model - g = 0
#---------------------

est_rob0 <- main_reg(y = data_rob[,1], X_dat = data_rob[,-1], vartype = c(1,1,0,0), 
                    niter = niter, nburn = nburn,
                    B0 = B0, m0 = m0, a0 = a0, b0 = b0, kappa = kappa, sigma = sigma, 
                    s2v = 5, g_fun = 0, thin = thin, lambda = 0, power = 100, 
                    pred = TRUE, Xnew_dat = Xnew, pred_grid = T, grid = grid)

ggplot(data.frame(x = grid, y = colMeans(est_rob0$prediction3), 
                  ylow = apply(est_rob0$prediction3, 2, quantile, p = 0.05), 
                  yup = apply(est_rob0$prediction3, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution") + 
  geom_vline(aes(xintercept = true_Eval), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.6))


# run the model - g = A
#---------------------

est_rob1 <- main_reg(y = data_rob[,1], X = data_rob[,-1], vartype = c(1,1,0,0), 
                    niter = niter, nburn = nburn,
                    B0 = B0, m0 = m0, a0 = a0, b0 = b0, kappa = kappa, sigma = sigma, 
                    s2v = 5, g_fun = 1, thin = thin, lambda = lambda_opt, power = 1,
                    pred = TRUE, Xnew_dat = Xnew, pred_grid = T, grid = grid)

ggplot(data.frame(x = grid, y = colMeans(est_rob1$prediction3), 
                  ylow = apply(est_rob1$prediction3, 2, quantile, p = 0.05), 
                  yup = apply(est_rob1$prediction3, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution") + 
  geom_vline(aes(xintercept = true_Eval), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, 0.6))

# run the model - g = C
#---------------------

est_rob3 <- main_reg(y = data_rob[,1], X = data_rob[,-1], vartype = c(1,1,0,0), 
                     niter = niter, nburn = nburn,
                     B0 = B0, m0 = m0, a0 = a0, b0 = b0, kappa = kappa, sigma = sigma, 
                     s2v = 5, g_fun = 3, thin = thin, lambda = lambda_opt, power = 1,
                     pred = T, Xnew_dat = Xnew, pred_grid = T, grid = grid)

ggplot(data.frame(x = grid, y = colMeans(est_rob3$prediction3), 
                  ylow = apply(est_rob3$prediction3, 2, quantile, p = 0.05), 
                  yup = apply(est_rob3$prediction3, 2, quantile, p = 0.95))) +
  theme_bw() +
  geom_line(aes(x = x, y = y)) + 
  xlab("y") +
  ylab("density") +
  ggtitle("Predictive distribution") + 
  geom_vline(aes(xintercept = true_Eval), lty = 2, col = 2) +
  geom_ribbon(aes(x = x, ymin = ylow, ymax = yup), alpha = 0.25) +
  ylim(c(0, .6))

#-------------- MSE REGRESSION --------------
#--------------------------------------------
# Data from Muller et al 2011
# same strategy for comparison
# 

data <- read.table("dtasim.txt", head = T)

#--------------------------------------------
# parameters

niter <- 15000
nburn <- 10000
thin <- 1

M <- 2
B0 <- diag(100, 4)
m0 <- rep(0, 4)
a0 <- 2
b0 <- 1
kappa <- 0.1
sigma <- 0.2

#--------------------------------------------

X_star <- rbind(c(-1,0,0), c(0,0,0), c(1,0,0),
                c(-1,1,0), c(0,1,0), c(1,1,0),
                c(-1,0,1), c(0,0,1), c(1,0,1),
                c(-1,1,1), c(0,1,1), c(1,1,1))

Y_star <- rep(0, 12)
pdf <- matrix(scan("py-p.mdp"), byrow=T, ncol=50)
y_grid <- scan("py-y.mdp")
for(h in 1:12){
  Y_star[h] <- sum(y_grid * pdf[h,]) / sum(pdf[h,])
}
Y_star

#--------------------------------------------

neff <- (niter - nburn) / thin
Y_pred <- matrix(0, ncol = nrow(X_star), nrow = neff)
RMSE_all <- matrix(0, ncol = nrow(X_star), nrow = M)

lambda_temp <- MC_routine(Y = as.matrix(data[,-1]), nrep = 10000, type_of_var = c(1,0,0))
lambda_opt <- 0.1 / mean(lambda_temp[,1])

#--------------------------------------------
set.seed(1234)
for(i in 1:M){
  current_idx <- sample(x = 1:1000, size = 200, replace = F)
  temp_data <- data[current_idx, ]
  est_model <- main_reg(y = temp_data[,1], X_dat = as.matrix(temp_data[,-1]), vartype = c(1,0,0),
                       niter = niter, nburn = nburn,
                       B0 = B0, m0 = m0, a0 = a0, b0 = b0, kappa = kappa, sigma = sigma,
                       s2v = 5, g_fun = 3, thin = thin, lambda = lambda_opt, power = 1,
                       pred = TRUE, Xnew_dat = X_star, pred_grid = T, grid = y_grid)

  RMSE_all[i,] <- round(sqrt((colMeans(est_model$prediction2) - Y_star)^2), digits = 2)
}

round(colMeans(RMSE_all), digits = 1)
round(mean(colMeans(RMSE_all)), digits = 1)

