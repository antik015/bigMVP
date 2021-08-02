### Example run ###

n = 200
p = 5
q = 50

## Generate data ##
X = matrix(rnorm(n*p), n, p)
rhoT = 0.5
SigT = (1-rhoT)*matrix(1, q, q) + rhoT*diag(q)
chol_SigT = chol(SigT)
betaT = matrix(rnorm((p+1)*q), p+1, q)
Z = cbind(1, X)%*%betaT + matrix(rnorm(n*q), n, q)%*%chol_SigT
Y = matrix(0, n, q)
Y[Z>0] = 1

### Hyperparameters ###
eta0 = rep(0, p+1)
nu0 = 0.001
gamma0 = p+1
Lambda0 = diag(p+1)
max_it = 100
epsilon = 0.0001
prior_var = 25
m = 15
nmcmc = 200
burnin = 50
K = p+1

### Source Cpp files ###
library(Rcpp)
sourceCpp("two_stage_parallel.cpp")

## Run ##
res1 = two_stage_probit_h(Y, X, eta0, nu0, gamma0, Lambda0, max_it, epsilon, m, nmcmc, burnin) ## Hierarchical version
res2 = two_stage_probit_nh(Y, X, prior_var, max_it, epsilon, m) ## Non-hierarchical version

plot(as.vector(betaT), as.vector(res1$coefficients), xlab = "True coefficients", ylab = "Estimated coefficients", main = "Results for hierarcical version")
hist(as.vector(res1$post_mean[upper.tri(SigT)]), xlab = "Estimated correlations", main = "All pairwise correlations = 0.5")

