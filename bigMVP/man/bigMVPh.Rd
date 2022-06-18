\name{bigMVPh}
\alias{bigMVPh}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fast approximations to the marginal posterior distributions of regression coefficients and correlation coefficients under a multivariate model.
}
\description{
The function implements bigMVP method from Chakraborty et. al.(2022+) for approximating marginal distributions of parameters under a multivariate probit model. A hierarchical prior on the regression coefficients and the correlation coefficients is used here.
It is especially useful when many of the binary outcomes are rare. Please see the reference for more details.
}
\usage{
bigMVP(Y, X, eta0, nu0, gamma0, Lambda0, eta0_rho, nu0_rho, gamma0_rho, lambda0_sq_rho, max_it, epsilon, m, nmcmc, burnin)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
 \item{Y}{ A n by q matrix of binary outcomes. Each element of this matrix is either 0 or 1.}
  \item{X}{ A n by p matrix of covariates. Covariates are expected to continuous. Categorical coavriates can be accommodated by modifying this matrix from the user's end using dummy variables.}
  \item{eta0}{ Mean of Normal Inverse Wishart prior on regression coefficients. Usually eta0 is vector zeros of appropriate dimensions. }
  \item{nu0}{ Precision factor of Normal Inverse Wishart prior on regression coefficients. Usually nu0 = 0.001. }
  \item{gamma0}{ Degrees of freedom of the Normal Inverse Wishart prior on regression coefficients. Usually gamma0 = p+1. }
  \item{Lambda0}{ Scale matrix of the Normal Inverse Wishart prior on regression coefficients. Usually Lambda0 = diag(p+1). }
  \item{eta0_rho}{ Mean of Normal Inverse Gamma prior on (transformed) correlation coefficients. Usually eta0_rho = 0.  }
  \item{nu0_rho}{ Precision factor of Normal Inverse Gamma prior on (transformed) correlation coefficients. Usually eta0_rho = 0.01. }
  \item{gamma0_rho}{ Shape of Normal Inverse Gamma prior on (transformed) correlation coefficients. Usually gamma0_rho = 0.01. }
  \item{lambda0_sq_rho}{ Scale of Normal Inverse Gamma prior on (transformed) correlation coefficients. Usually lambda0_sq_rho = 0.01.  }
  \item{max_it}{ Maximum iterations allowed for Laplace approximations. A good default choice is 100.}
  \item{epsilon}{ Convergence criteria for Laplace approximation. A good default choice is 0.0001.}
  \item{m}{Number of quadrature points for Gauss-Legendre numerical integration. A good default choice is 15.}
  \item{nmcmc}{Number of MCMC iterations for approximate sampling. Usually set to 1000.}
  \item{burnin}{Number of MCMC iterations for burnin for approximate sampling. Usually set to 200.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
\item{coefficients}{A (p+1) by q matrix. Each column is the mean of Gaussian approximation to the marginal posterior distribution of the p covariates on the q binary outcomes including the intercept.}
\item{cov_mats}{A (p+1)^2 by q matrix. Each column is covariance matrix of Gaussian approximation to the marginal posterior distribution of the p covariates on the q binary outcomes including the intercept.}
\item{post_mean}{A q by q matrix. Each off-diagonal element is the mean of the approximation Gaussian distribution of the marginal posterior of the corresponding correlation coefficient. Diagonal elements are 1. Not necessarily a positive definite matrix.}
\item{post_var}{A q by q matrix. Each off-diagonal element is the variance of the approximation Gaussian distribution of the marginal posterior of the corresponding correlation coefficient. Diagonal elements are 1.}
\item{runtime}{Runtime in seconds.}
}
\references{
Chakraborty, Antik, Rihui Ou, and David B. Dunson. "Bayesian inference on high-dimensional multivariate binary data." arXiv preprint arXiv:2106.02127 (2021).
}
\author{
Antik Chakraborty
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
n = 200
p = 2
q = 4

## Generate data ##
X = matrix(rnorm(n*p), n, p)
rhoT = 0.5
SigT = (1-rhoT)*matrix(1, q, q) + rhoT*diag(q)
chol_SigT = chol(SigT)
betaT = matrix(rnorm((p+1)*q), p+1, q)
Z_T = cbind(1, X)%*%betaT + matrix(rnorm(n*q), n, q)%*%chol_SigT
Y = matrix(0, n, q)
Y[Z_T>0] = 1

## Hyperparameters ##
prior_var = 25
max_it = 100
epsilon = 0.0001
m = 15
eta0 = rep(0, p+1)
nu0 = 0.001
gamma0 = p+1
Lambda0 = diag(p+1)
eta0_rho = 0
nu0_rho = 0.01
gamma0_rho = 0.01
lambda0_sq_rho = 0.01
nmcmc = 1000
burnin = 200

## Example run ##
res1 = bigMVPh(Y, X, eta0, nu0, gamma0, Lambda0, eta0_rho, nu0_rho, gamma0_rho, lambda0_sq_rho, max_it, epsilon, m, nmcmc, burnin)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
