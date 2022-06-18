\name{mvp_mcmc_ni}
\alias{mvp_mcmc_ni}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
MCMC implementation of the multivariate probit model with MNI(nu) prior on the correlation matrix from Huang and Wand(2013).
}
\description{
This function implements a standard MCMC implementation of the multivariate probit model with Gaussian prior on the regression coefficients and MNI(nu) prior on the correlation matrix. 
}
\usage{
mvp_mcmc_ni(Y, X, nu, nmcmc, burnin, prior_var)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{ A n by q matrix of binary outcomes. Each element of this matrix is either 0 or 1.}
  \item{X}{ A n by p matrix of covariates. Covariates are expected to continuous. Categorical coavriates can be accommodated by modifying this matrix from the user's end using dummy variables.}
  \item{nu}{Parameter nu of the MNI distribution. nu = 2 implies a uniform prior distribution over correlation matrices.}
  \item{prior_var}{ Prior variance of the Gaussian prior on the regression coefficients. The mean is set as 0. A good default choice is 25.}
  \item{nmcmc}{Number of MCMC iterations for approximate sampling. }
  \item{burnin}{Number of MCMC iterations for burnin for approximate sampling.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
\item{B_samples}{A (p+1) by q by (nmcmc - burnin) sized array. B_samples[,,j] stores the jth MCMC sample of the regression coefficients including the intercept.}
\item{S_samples}{A q by q by (nmcmc - burnin) sized array. S_samples[,,j] stores the jth MCMC sample of the correlation matrix.}
}
\references{
Chakraborty, Antik, Rihui Ou, and David B. Dunson. "Bayesian inference on high-dimensional multivariate binary data." arXiv preprint arXiv:2106.02127 (2021).
Huang, Alan, and Matthew P. Wand. "Simple marginally noninformative prior distributions for covariance matrices." Bayesian Analysis 8.2 (2013): 439-452.
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
nu = 1
prior_var = 25
nmcmc = 1000
burnin = 200

## res_mcmc_ni = mvp_mcmc_ni(Y, X, nu, nmcmc, burnin, prior_var)
}% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
