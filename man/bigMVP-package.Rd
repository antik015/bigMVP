\name{bigMVP-package}
\alias{bigMVP-package}
\docType{package}
\title{
\packageTitle{bigMVP}
}
\description{
\packageDescription{bigMVP}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{bigMVP}
\packageIndices{bigMVP}
Consider the model $z = B'x+ \epsilon$, where $\epsilon \sim \text{N}(0, \Sigma)$. The observed data is $y_{ij} = 1(z_{ij} >0)$, with $i = 1, \ldots, n$ and $j = 1, \ldots, q$. Here $X$ is a $n \times p$ matrix of covariates. The above model is known as the multivariate model where $B$ is the matrix of regression coefficients and $\Sigma$ is a correlation matrix (diagonal elements are equal to 1).
The package implements functions bigMVP_uni(), bigMVP_lkj(), bigMVPh(), mvp_mcmc_lkj() and mvp_mcmc_ni(). The first three functions provide fast approximations to the marginal posterior distribution of the regression coefficients and correlation coefficients under a multivariate probit model. The functions mvp_mcmc_lkj() and mvp_mcmc_ni() implements an MCMC approximation to the full posterior distribution of the aforementioned parameters. The MCMC implementation is not scalable when the number of binary responses is high, but is provided as a benchmark for the fast approximations.
}
\author{
\packageAuthor{bigMVP}

Maintainer: \packageMaintainer{bigMVP}
}
\references{
Chakraborty, Antik, Rihui Ou, and David B. Dunson. "Bayesian inference on high-dimensional multivariate binary data." arXiv preprint arXiv:2106.02127 (2021).
}

\keyword{ package }
\seealso{
~~ Optional links to other man pages, e.g. ~~
~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
## Data dimensions ##
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

## Example run ##
res1 = bigMVP(Y, X, prior_var, max_it, epsilon, m)
}
