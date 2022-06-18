mvp_mcmc_lkj = function(Y, X, nu, nmcmc, burnin, prior_var)
{
  ## Data dimensions
  n = nrow(X)
  p = ncol(X)
  q = ncol(Y)
  #X = cbind(1, X)
  ## Contsraint matrices
  # mat_list = constraint_mat(Y)
  # g = rep(0, q)
  L = matrix(0, n, q)
  U = matrix(0, n, q)
  for(i in 1:n)
  {
    for(j in 1:q)
    {
      if(Y[i,j] == 1)
      {
        U[i,j] = Inf
      } else if(Y[i,j] == 0)
      {
        L[i,j] = -Inf
      }
    }
  }
  
  ## Initial values 
  B = matrix(0, p+1, q)
  S = diag(q)
  #S = SigT
  Z = matrix(0, n, q)
  # Z[Y==1] = 0.5
  # Z[Y==0] = -0.5
  
  ## Other definitions 
  X_tilde = kronecker(diag(q), cbind(1,X))
  S_inv = chol2inv(chol(S))
  S_tilde_inv = kronecker(chol2inv(chol(S)), diag(n))
  
  ## Storage
  #Bout = list()
  #Sout = list()
  Bout = array(0, dim = c(p+1, q, nmcmc - burnin))
  Sout = array(0, dim = c(q, q, nmcmc - burnin))
  for(ii in 1:nmcmc)
  {
    ## Update latent variables
    for(i in 1:n)
    {
      # r = t(t(t(B)%*%c(1,X[i,]))%*%chol2inv(chol(S)))
      # Z[i,] = rtmg(1, M = S, initial = Z[i,], r = as.vector(r), f = mat_list[[i]], g = as.vector(g))
      Z[i,] = tmvnsim(1, q, L[i,], U[i,], means = t(B)%*%c(1, X[i,]), sigma = S)$samp
    }
    
    ## Update regression coefficients
    Q = t(X_tilde)%*%S_tilde_inv%*%X_tilde + (1/prior_var)*diag((p+1)*q)
    post_cov = chol2inv(chol(Q))
    post_mean = post_cov%*%t(X_tilde)%*%S_tilde_inv%*%as.vector(Z)
    beta = rmvnorm(1, post_mean, post_cov)
    B = matrix(beta, p+1, q)
    
    ##print(ii)
    ## Update correlation matrix
    
    S = riwish(n - 2*nu - q + 1, t(Z - cbind(1,X)%*%B)%*%(Z - cbind(1,X)%*%B))
    S_inv = chol2inv(chol(S))
    S_tilde_inv = kronecker(chol2inv(chol(S)), diag(n))
    
    
    
    if(ii>burnin)
    {
      l = diag(S)
      L1 = diag(l^(-1/2))
      Sout[,, ii- burnin] = cov2cor(S)
      Bout[, , ii-burnin] = B%*%L1
    }
    if(ii%%100 ==0)
    {
      print(ii)
    }
  }
  result = list("B_samples" = Bout, "S_samples" = Sout)
}

mvp_mcmc_ni = function(Y, X, nu, nmcmc, burnin, prior_var)
{
  ## Data dimensions
  n = nrow(X)
  p = ncol(X)
  q = ncol(Y)
  
  ## Hyperparameter values 
  A = rep(10^4, q)
  
  
  L = matrix(0, n, q)
  U = matrix(0, n, q)
  for(i in 1:n)
  {
    for(j in 1:q)
    {
      if(Y[i,j] == 1)
      {
        U[i,j] = Inf
      } else if(Y[i,j] == 0)
      {
        L[i,j] = -Inf
      }
    }
  }
  
  ## Initial values 
  B = matrix(0, p+1, q)
  S = diag(q)
  a = rep(1, q)
  #S = SigT
  Z = matrix(0, n, q)
  # Z[Y==1] = 0.5
  # Z[Y==0] = -0.5
  
  ## Other definitions 
  X_tilde = kronecker(diag(q), cbind(1,X))
  S_inv = chol2inv(chol(S))
  S_tilde_inv = kronecker(S_inv, diag(n))
  
  ## Storage
  #Bout = list()
  #Sout = list()
  Bout = array(0, dim = c(p+1, q, nmcmc - burnin))
  Sout = array(0, dim = c(q, q, nmcmc - burnin))
  for(ii in 1:nmcmc)
  {
    ## Update latent variables
    for(i in 1:n)
    {
      # r = t(t(t(B)%*%c(1,X[i,]))%*%chol2inv(chol(S)))
      # Z[i,] = rtmg(1, M = S, initial = Z[i,], r = as.vector(r), f = mat_list[[i]], g = as.vector(g))
      Z[i,] = tmvnsim(1, q, L[i,], U[i,], means = t(B)%*%c(1, X[i,]), sigma = S)$samp
    }
    
    ## Update regression coefficients
    Q = t(X_tilde)%*%S_tilde_inv%*%X_tilde + (1/prior_var)*diag((p+1)*q)
    post_cov = chol2inv(chol(Q))
    post_mean = post_cov%*%t(X_tilde)%*%S_tilde_inv%*%as.vector(Z)
    beta = rmvnorm(1, post_mean, post_cov)
    B = matrix(beta, p+1, q)
    
    ##print(ii)
    ## Update correlation matrix
    
    S = riwish(nu+ q + n - 1, t(Z - cbind(1,X)%*%B)%*%(Z - cbind(1,X)%*%B))
    S_inv = chol2inv(chol(S))
    S_tilde_inv = kronecker(S_inv, diag(n))
    
    
    ## Update a ##
    for(j in 1:q)
    {
      a[j] = rinvgamma(1,0.5*(nu+q), nu*S_inv[j,j] + 1/A[j])
    }
    if(ii>burnin)
    {
      l = diag(S)
      L1 = diag(l^(-1/2))
      #Sout[[ii - burnin]] = cov2cor(S)
      Sout[,, ii- burnin] = cov2cor(S)
      Bout[, , ii-burnin] = B%*%L1
    }
    if(ii%%100 ==0)
    {
      print(ii)
    }
  }
  result = list("B_samples" = Bout, "S_samples" = Sout)
}
