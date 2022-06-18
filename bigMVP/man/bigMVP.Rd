\name{bigMVP}
\alias{bigMVP}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fast approximations to the marginal posterior distributions of regression coefficients and correlation coefficients under a multivariate model.
}
\description{
The function implements bigMVP method from Chakraborty et. al.(2022+) for approximating marginal distributions of parameters under a multivariate probit model. Prior on the regerssion coefficient vector for each binary outcome is a zero mean Gaussian with variance 'prior_var' and the prior on each correlation coefficient is Uniform(-1,1).
It is especially useful when the number of binary responses is very high. Note that, approximations to the joint posterior is not provided.
}
\usage{
bigMVP(Y, X, prior_var, max_it, epsilon, m)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{ A n by q matrix of binary outcomes. Each element of this matrix is either 0 or 1.}
  \item{X}{ A n by p matrix of covariates. Covariates are expected to continuous. Categorical coavriates can be accommodated by modifying this matrix from the user's end using dummy variables.}
  \item{prior_var}{ Prior variance of the Gaussian prior on the regression coefficients. The mean is set as 0. A good default choice is 25.}
  \item{max_it}{ Maximum iterations allowed for Laplace approximations. A good default choice is 100.}
  \item{epsilon}{ Convergence criteria for Laplace approximation. A good default choice is 0.0001.}
  \item{m}{Number of quadrature points for Gauss-Legendre numerical integration. A good default choice is 15.}
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

%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
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

## Example run ##
res1 = bigMVP(Y, X, prior_var, max_it, epsilon, m)
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
