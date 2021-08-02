#include <RcppArmadillo.h>
#include <omp.h>
// [[Rcpp::depends(RcppArmadillo)]]


using namespace Rcpp;



// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
  //
  //   http://www.rcpp.org/
  //   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
  //
  // [[Rcpp::export]]

double bvn(double h, double k, double r) {
  NumericVector w;
  NumericVector x;
  if (abs(r)<0.3){ 
    w = {0.171324492379170,   0.360761573048138,   0.467913934572690,   0.171324492379170,   0.360761573048138,   0.467913934572690};
    x = {0.067530485796848,   0.338790613533735,   0.761380813916803,   1.932469514203152,   1.661209386466265,   1.238619186083197};
  }
  else if (abs(r)<0.75){
    w = {0.047175336386512,   0.106939325995318,   0.160078328543346,   0.203167426723066,   0.233492536538355,   0.249147045813403,   0.047175336386512, 0.106939325995318,   0.160078328543346,  0.203167426723066,   0.233492536538355,  0.249147045813403};
    x = {0.018439365753281,   0.095882743629525,   0.230097325805695,   0.412682045713383,   0.632168501001820,   0.874766591488531,   1.981560634246719, 1.904117256370475,   1.769902674194305,   1.587317954286617,   1.367831498998180,   1.125233408511469};
  }
  else{
    w = {0.017614007139152,   0.040601429800387,   0.062672048334109,   0.083276741576705,   0.101930119817240,   0.118194531961518,   0.131688638449177, 0.142096109318382,   0.149172986472604,   0.152753387130726,   0.017614007139152,   0.040601429800387,   0.062672048334109,   0.083276741576705, 0.101930119817240,   0.118194531961518,   0.131688638449177,   0.142096109318382,   0.149172986472604,   0.152753387130726};
    x = {0.006871400814905,   0.036028072722086,   0.087765571748674,   0.160883028177781,   0.253668093539849,   0.363946319273485,   0.489132998049173, 0.626293911284580,   0.772214148858355,   0.923473478866503,   1.993128599185095,   1.963971927277914,   1.912234428251326,   1.839116971822219, 1.746331906460151,   1.636053680726515,   1.510867001950827,   1.373706088715420,   1.227785851141645,   1.076526521133497};
  }
  double tp = 2 * M_PI ;
  double hk = h * k;
  double hs = (h*h + k*k)/2;
  double asr = asin(r)/2;
  NumericVector sn = sin(asr*x);
  NumericVector bvn = exp((sn*hk-hs)/(1-pow(sn,2)));
  double prob = std::inner_product(bvn.begin(), bvn.end(), w.begin(), 0.0);
  prob = prob * asr / tp + 0.25 * erfc(h/sqrt(2)) * erfc(k/sqrt(2));
  return prob;
}

// [[Rcpp::export]]
double cox_bvn0(double a, double b, double rho){
  double l1;
  if(rho>0){
    double l = 0.0;
    if(a>=0 || b>=0){
      double c = std::max(a,b);
      double d = std::min(a,b);
      double phi_c = arma::normcdf(-c);
      double mu_c = arma::normpdf(c)/phi_c;
      double xi = (rho*mu_c - d)/pow(1 - rho*rho, 0.5);
      l = phi_c*arma::normcdf(xi) - 1 + arma::normcdf(c) + arma::normcdf(d);
    } else if(a<0 && b<0){
      double c = std::max(-a,-b);
      double d = std::min(-a,-b);
      double phi_c = arma::normcdf(-c);
      double mu_c = arma::normpdf(c)/phi_c;
      double xi = (rho*mu_c - d)/pow(1 - rho*rho, 0.5);
      l = phi_c*arma::normcdf(xi);
    }
    l1 = l;
  } else {
    double l;
    l = arma::normcdf(a) - cox_bvn0(a, -b, -rho);
    l1 = l;
  }
  return l1;
}

// [[Rcpp::export]]
arma::vec cox_bvn(arma::vec a, arma::vec b, arma::vec rhos){
  int n = rhos.size();
  arma::vec l(n, arma::fill::zeros);
  for(int i=0; i<n; ++i){
    l(i) = log(cox_bvn0(a(i), b(i), rhos(i)));
  }
  return l;
}


// [[Rcpp::export]]
arma::vec mat_diag(arma::mat A, arma::mat B, arma::mat C){
  int n = A.n_rows;
  int p = A.n_cols;
  arma::vec d(n, arma::fill::zeros);
  arma::mat D = A*B;
  for(int i=0; i<n; ++i){
    d(i) = sum(D(i, arma::span(0,p-1)).t()%C(arma::span(0,p-1),i));
  }
  return(d);
}


// [[Rcpp::export]]

arma::mat vec_log_post_beta_d1_d2_nh(arma::vec y, arma::mat X, arma::vec beta, double prior_var){
  int n = y.size();
  int p = X.n_cols;
  arma::vec q = 2*y - arma::ones(n);
  arma::vec lambda(n, arma::fill::zeros);
  for(int i=0; i<n; ++i){
    lambda(i) = q(i)*arma::normpdf(q(i)*sum(X(i,arma::span(0,p-1))*beta))/arma::normcdf(q(i)*sum(X(i,arma::span(0,p-1))*beta));
  }
  arma::vec del_log_posterior(p, arma::fill::zeros);
  for(int j=0; j<p; ++j){
    del_log_posterior(j) = sum(lambda.t()*X(arma::span(0,n-1), j));
  }
  arma::vec score = del_log_posterior - beta/prior_var;
  
  arma::mat hessian(p, p, arma::fill::zeros);
  for(int i=0; i<n; ++i){
    hessian = hessian + lambda(i)*((sum(X(i,arma::span(0,p-1))*beta)) + lambda(i))*X(i, arma::span(0, p-1)).t()*X(i, arma::span(0,p-1));
  }
  arma::mat I = arma::diagmat(arma::ones(p));
  hessian = hessian + (1/prior_var)*I;
  arma::mat result(p, p+1);
  result(arma::span(0, p-1), 0) = score;
  result(arma::span(0, p -1), arma::span(1, p)) = -hessian;
  return result;
}

// [[Rcpp::export]]

arma::mat vec_log_post_beta_d1_d2_h(arma::vec y, arma::mat X, arma::vec beta, arma::vec eta, arma::mat Omega){
  int n = y.size();
  int p = X.n_cols;
  arma::vec q = 2*y - arma::ones(n);
  arma::vec lambda(n, arma::fill::zeros);
  for(int i=0; i<n; ++i){
    lambda(i) = q(i)*arma::normpdf(q(i)*sum(X(i,arma::span(0,p-1))*beta))/arma::normcdf(q(i)*sum(X(i,arma::span(0,p-1))*beta));
  }
  arma::vec del_log_posterior(p, arma::fill::zeros);
  for(int j=0; j<p; ++j){
    del_log_posterior(j) = sum(lambda.t()*X(arma::span(0,n-1), j));
  }
  arma::mat I = diagmat(arma::ones(p));
  arma::mat Omega_inv = solve(Omega, I);
  arma::vec score = del_log_posterior - Omega_inv*(beta - eta);
  
  arma::mat hessian(p, p, arma::fill::zeros);
  for(int i=0; i<n; ++i){
    hessian = hessian + lambda(i)*((sum(X(i,arma::span(0,p-1))*beta)) + lambda(i))*X(i, arma::span(0, p-1)).t()*X(i, arma::span(0,p-1));
  }
  hessian = hessian + Omega_inv;
  arma::mat result(p, p+1);
  result(arma::span(0, p-1), 0) = score;
  result(arma::span(0, p -1), arma::span(1, p)) = -hessian;
  return result;
}

// [[Rcpp::export]]

arma::vec vec_log_post_beta_laplace_nh(arma::vec y, arma::mat X, double prior_var, int max_it, double epsilon){
  
  // change so that it returns a vector
  int n = y.size();
  arma::mat X_new = join_rows(arma::ones(n, 1), X);
  int p = X.n_cols;
  arma::vec b(p+1, arma::fill::zeros);
  arma::mat H(p+1, p+1, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = vec_log_post_beta_d1_d2_nh(y, X_new, b, prior_var);
    arma::vec u = res(arma::span(0, p), 0);
    H = res(arma::span(0, p), arma::span(1, p+1));
    arma::vec err = -inv(H)*u;
    double norm_err = norm(err, 2);
    if(norm_err<epsilon){
      break;
    }
    b = b - inv(H)*u;
  }
  double vec_size = (p+1) + (p+1)*(p+1) + 3*n;
  arma::vec result(vec_size);
  result(arma::span(0, p)) = b;
  arma::mat I = arma::diagmat(arma::ones(p+1));
  arma::mat H_inv = solve(-H, I);
  arma::vec cov_vec = arma::vectorise(H_inv);
  result(arma::span(p+1, p + pow(p+1, 2))) = cov_vec;
  arma::vec s = mat_diag(X_new, H_inv, X_new.t());
  result(arma::span(p + pow(p+1, 2) + 1, p + pow(p+1, 2) + n)) = arma::ones(n) + s;
  arma::vec q = 2*y - arma::ones(n);
  arma::vec m = X_new*b/pow(arma::ones(n) + s, 0.5);
  result(arma::span(p + pow(p+1, 2)+ n + 1, p + pow(p+1, 2) + 2*n)) = q;
  result(arma::span(p + pow(p+1, 2)+ 2*n + 1, p + pow(p+1, 2) + 3*n)) = q%m;
  return result;
}

// [[Rcpp::export]]

arma::vec vec_log_post_beta_laplace_h(arma::vec y, arma::mat X, arma::vec eta, arma::mat Omega, int max_it, double epsilon){
  
  // change so that it returns a vector
  int n = y.size();
  arma::mat X_new = join_rows(arma::ones(n, 1), X);
  int p = X.n_cols;
  arma::vec b(p+1, arma::fill::zeros);
  arma::mat H(p+1, p+1, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = vec_log_post_beta_d1_d2_h(y, X_new, b, eta, Omega);
    arma::vec u = res(arma::span(0, p), 0);
    H = res(arma::span(0, p), arma::span(1, p+1));
    arma::vec err = -inv(H)*u;
    double norm_err = norm(err, 2);
    if(norm_err<epsilon){
      break;
    }
    b = b - inv(H)*u;
  }
  double vec_size = (p+1) + (p+1)*(p+1) + 3*n;
  arma::vec result(vec_size);
  result(arma::span(0, p)) = b;
  arma::mat I = arma::diagmat(arma::ones(p+1));
  arma::mat H_inv = solve(-H, I);
  arma::vec cov_vec = arma::vectorise(H_inv);
  result(arma::span(p+1, p + pow(p+1, 2))) = cov_vec;
  arma::vec s = mat_diag(X_new, H_inv, X_new.t());
  result(arma::span(p + pow(p+1, 2) + 1, p + pow(p+1, 2) + n)) = arma::ones(n) + s;
  arma::vec q = 2*y - arma::ones(n);
  arma::vec m = X_new*b/pow(arma::ones(n) + s, 0.5);
  result(arma::span(p + pow(p+1, 2)+ n + 1, p + pow(p+1, 2) + 2*n)) = q;
  result(arma::span(p + pow(p+1, 2)+ 2*n + 1, p + pow(p+1, 2) + 3*n)) = q%m;
  return result;
}


// [[Rcpp::export]]

arma::mat marginal_probit_nh(arma::mat Y, arma::mat X, double prior_var, double epsilon, int max_it){
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  double vec_size = (p+1) + (p+1)*(p+1) + 3*n;
  arma::mat all_res(vec_size, q);
  //omp_set_dynamic(0);
  omp_set_num_threads(4);
  #pragma omp parallel for 
  for(int j=0; j<q; ++j){
    all_res(arma::span(0, vec_size - 1), j) = vec_log_post_beta_laplace_nh(Y(arma::span(0,n-1), j), X, prior_var, max_it, epsilon); 
  }
  return all_res;
}

// [[Rcpp::export]]

arma::mat marginal_probit_h(arma::mat Y, arma::mat X, arma::vec eta, arma::mat Omega, double epsilon, int max_it){
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  double vec_size = (p+1) + (p+1)*(p+1) + 3*n;
  arma::mat all_res(vec_size, q);
  //omp_set_dynamic(0);
  omp_set_num_threads(4);
  #pragma omp parallel for 
  for(int j=0; j<q; ++j){
    all_res(arma::span(0, vec_size - 1), j) = vec_log_post_beta_laplace_h(Y(arma::span(0,n-1), j), X, eta, Omega, max_it, epsilon); 
  }
  return all_res;
}

// [[Rcpp::export]]
arma::mat gen_mvnrnd(arma::mat M, arma::mat Covs){
  int p = M.n_rows;
  int q = M.n_cols;
  arma::mat X(p,q);
  for(int j=0; j<q; ++j)
  {
    arma::vec m = M(arma::span(0,p-1), j);
    arma::mat S = reshape(Covs(arma::span(0, p*p - 1), j), p, p);
    X(arma::span(0,p-1),j) = arma::mvnrnd(m, S);
  }
  return X;
}

// [[Rcpp::export]]
arma::mat list_mean_cpp(List L){
  arma::mat L0 = L[0];
  int p = L0.n_rows;
  int q = L0.n_cols;
  int n = L.length();
  arma::mat L_mean(p, q);
  for(int i=0; i<n; ++i){
    arma::mat L_i = L[i];
    L_mean = L_mean + L_i;
  }
  L_mean = L_mean/n;
  return L_mean;
}

// [[Rcpp::export]]

arma::mat first_stage_sampling(arma::mat Y, arma::mat X, arma::vec eta0, double nu0, double gamma0, arma::mat Lambda0, int max_it, double epsilon, double nmcmc, double burnin){
  int q = Y.n_cols;
  int p = X.n_cols;
  List beta_samples((nmcmc - burnin));
  arma::mat eta_samples(p+1, (nmcmc - burnin));
  List Omega_samples((nmcmc - burnin));
  arma::vec eta = arma::zeros(p+1);
  arma::mat Omega = diagmat(arma::ones(p+1));
  double gamma_q = gamma0 + q;
  double nu_q = nu0 + q;
  
  // Start sampling //
  for(int ii =1; ii<=nmcmc; ++ii){
    arma::mat beta_res = marginal_probit_h(Y, X, eta, Omega, epsilon, max_it);
    arma::mat beta_mean = beta_res(arma::span(0, p), arma::span(0, q-1));
    arma::mat cov_mats = beta_res(arma::span(p+1, p + pow(p+1, 2)), arma::span(0, q-1));
    // check columns with NA results
    arma::mat b_samples = gen_mvnrnd(beta_mean, cov_mats);
    arma::vec beta_bar = arma::mean(b_samples, 1);
    arma::mat beta_cov = (q-1)*arma::cov(b_samples.t());
    arma::mat Lambda_q = Lambda0 + beta_cov + (nu0*q/(nu0 + q))*(beta_bar - eta0)*(beta_bar - eta0).t();
    //Lambda_q.print();
    Omega = arma::iwishrnd(Lambda_q, gamma_q);
    arma::vec beta1 = (nu0*eta0 + q*beta_bar)/nu_q;
    //Omega.print();
    eta = arma::mvnrnd(beta1, Omega/nu_q);
    
    if(ii> burnin)
    {
      beta_samples[(ii - burnin - 1)] = b_samples;
      eta_samples(arma::span(0, p), (ii - burnin - 1)) = eta;
      Omega_samples[(ii - burnin - 1)] = Omega;
    }
    
    if(ii%100 == 0)
    {
      Rprintf("Iteration = %d \n", ii);
    }
  }
  arma::vec eta_hat = arma::mean(eta_samples, 1);
  arma::mat Omega_hat = list_mean_cpp(Omega_samples);
  Omega_hat = 0.5*(Omega_hat + Omega_hat.t());
  arma::mat result = join_rows(eta_hat, Omega_hat);
  return result;
}

// [[Rcpp::export]]
double post_cor_new(double rho, arma::vec q1, arma::vec q2, arma::vec m1, arma::vec m2, arma::vec s1, arma::vec s2){
  int n = m1.size();
  arma::vec rho_all = q1(arma::span(0,n-1))%q2(arma::span(0, n-1))%(rho*arma::ones(n))/(pow(s1, 0.5)%pow(s2, 0.5));
  arma::vec l = cox_bvn(m1(arma::span(0,n-1)), m2(arma::span(0,n-1)), rho_all);
  return sum(l);
}

// [[Rcpp::export]]
List GLRcpp(int n, double a, double b){
  
  // You might see const references in production code often...
  
  const arma::vec& i = arma::linspace(1, n-1, n-1);
  
  arma::mat J(n,n,arma::fill::zeros);
  
  // Beta diagonal from Golub-Welsch
  
  const arma::vec& d = i / (arma::sqrt(4 * arma::square(i) - 1));
  
  // Setting off-diagonal elements
  
  J.diag(1) = d;
  J.diag(-1) = d;
  
  // Initialize matrix and vector for eigenvalues/vectors
  
  arma::vec L;
  arma::mat V;
  
  arma::eig_sym(L, V, J);
  
  // Only need first row...
  
  arma::vec w = 2 * arma::vectorise(arma::square(V.row(0)));
  
  arma::vec x = .5 * ((b - a) * L + a + b);
  w = -.5 * (a - b) * w;
  
  return List::create(
    Named("x") = x,
    Named("w") = w);
  
}


// [[Rcpp::export]]
List marginal_pairwise_probit(arma::mat params, int m, arma::mat Y, arma::mat X){
  arma::wall_clock timer;
  timer.tic();
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  List GL_res = GLRcpp(m, -1, 1);
  arma::vec rhos = GL_res[0];
  arma::vec wts = GL_res[1];
  arma::mat beta = params(arma::span(0, p), arma::span(0, q-1));
  //List cov_mats = params[1];
  arma::mat S = params(arma::span(p + pow(p+1, 2) + 1, p + pow(p+1, 2) + n), arma::span(0, q-1));
  arma::mat Q = params(arma::span(p + pow(p+1, 2)+ n + 1, p + pow(p+1, 2) + 2*n), arma::span(0, q-1));
  arma::mat M = params(arma::span(p + pow(p+1, 2)+ 2*n + 1, p + pow(p+1, 2) + 3*n), arma::span(0, q-1));
  arma::mat corr_mean = arma::mat(q, q);
  arma::mat corr_var = arma::mat(q, q);
  corr_mean.diag() = arma::ones(q);
  corr_var.diag() = arma::zeros(q);
  //omp_set_dymanic(0);
  omp_set_num_threads(4);
  #pragma omp parallel for 
  for(int j=0; j<(q-1); ++j){
    for(int k=(j+1); k<q; ++k){
      arma::vec d(m);
      for(int l=0; l<m; ++l){
        d(l) = post_cor_new(rhos(l), Q(arma::span(0,n-1),j), Q(arma::span(0,n-1),k), M(arma::span(0,n-1),j), M(arma::span(0,n-1),k), S(arma::span(0,n-1),j), S(arma::span(0,n-1),k));
      }
      arma::uvec id = find_finite(d);
      double m_const = median(d(id));
      arma::vec d1 = exp(-m_const*arma::ones(id.size()) + d(id));
      corr_mean(j,k) = corr_mean(k,j) = sum(wts(id)%d1%rhos(id))/sum(wts(id)%d1);
      corr_var(j,k) = corr_var(k,j) = sum(wts(id)%d1%pow(rhos(id), 2))/sum(wts(id)%d1) - pow(corr_mean(j,k),2);
    }
  }
  List result;
  //arma::mat U(q,q);
  //arma::vec ev(q);
  //eig_sym(ev, U, corr_mean);
  //arma::vec proj_ev(q);
  //for(int j=0; j<q; ++j){
    //if(ev(j)<=0) proj_ev(j) = 0;
  //}
  //arma::mat proj_corr_mean = U*diagmat(proj_ev)*U.t();
  double runtime = timer.toc();
  result["post_mean"] = corr_mean;
  //result["proj_post_mean"] = proj_corr_mean;
  result["post_var"] = corr_var;
  result["runtime"] = runtime;
  return result;
}

// [[Rcpp::export]]

List two_stage_probit_nh(arma::mat Y, arma::mat X, double prior_var, int max_it, double epsilon, int m){
  int q = Y.n_cols;
  int p = X.n_cols;
  arma::wall_clock timer;
  timer.tic();
  arma::mat first_stage = marginal_probit_nh(Y, X, prior_var, epsilon, max_it);
  arma::mat cov_mats = first_stage(arma::span(p+1, p + pow(p+1, 2)), arma::span(0, q-1));
  List second_stage = marginal_pairwise_probit(first_stage, m, Y, X);
  List result;
  result["coefficients"] = first_stage(arma::span(0, p), arma::span(0, q-1));
  result["cov_mats"] = cov_mats;
  result["post_mean"] = second_stage[0];
  result["post_var"] = second_stage[1];
  double runtime = timer.toc();
  result["runtime"] = runtime;
  return result;
}

// [[Rcpp::export]]

List two_stage_probit_h(arma::mat Y, arma::mat X, arma::vec eta0, double nu0, double gamma0, arma::mat Lambda0, int max_it, double epsilon, int m, int nmcmc, int burnin){
  int q = Y.n_cols;
  int p = X.n_cols;
  arma::wall_clock timer;
  timer.tic();
  arma::mat emp_bayes_estimates = first_stage_sampling(Y, X, eta0, nu0, gamma0, Lambda0, max_it, epsilon, nmcmc, burnin);
  arma::vec eta = emp_bayes_estimates(arma::span(0, p), 0);
  arma::mat Omega = emp_bayes_estimates(arma::span(0, p), arma::span(1, p+1));
  arma::mat first_stage = marginal_probit_h(Y, X, eta, Omega, epsilon, max_it);
  arma::mat cov_mats = first_stage(arma::span(p+1, p + pow(p+1, 2)), arma::span(0, q-1));
  List second_stage = marginal_pairwise_probit(first_stage, m, Y, X);
  List result;
  result["coefficients"] = first_stage(arma::span(0, p), arma::span(0, q-1));
  result["cov_mats"] = cov_mats;
  result["post_mean"] = second_stage[0];
  result["post_var"] = second_stage[1];
  double runtime = timer.toc();
  result["runtime"] = runtime;
  return result;
}
