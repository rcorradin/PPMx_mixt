#include "RcppArmadillo.h"
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

/* 
  Compute the posterior similarity matrix
*/ 

//[[Rcpp::export]]
arma::mat psm(arma::mat M){
  // initialize results
  arma::mat result(M.n_cols, M.n_cols, arma::fill::zeros);
  
  for(arma::uword i = 0; i < M.n_cols; i++){
    for(arma::uword j = 0; j <= i; j++){
      result(i,j) = arma::accu(M.col(i) == M.col(j));
      result(j,i) = result(i,j);
    }
    Rcpp::checkUserInterrupt();
  }
  return(result / M.n_rows);
}

/* 
  Compute the variation of information lower bound
*/ 

//[[Rcpp::export]]
arma::vec VI_LB(arma::mat C_mat, arma::mat psm_mat){
  
  arma::vec result(C_mat.n_rows);
  double f = 0.0;
  int n = psm_mat.n_cols;
  arma::vec tvec(n);
  
  for(arma::uword j = 0; j < C_mat.n_rows; j++){
    f = 0.0;
    for(arma::uword i = 0; i < n; i++){
      tvec = psm_mat.col(i);
      f += (log2(arma::accu(C_mat.row(j) == C_mat(j,i))) +
        log2(arma::accu(tvec)) -
        2 * log2(arma::accu(tvec.elem(arma::find(C_mat.row(j).t() == C_mat(j,i))))))/n;
    }
    result(j) = f;
    Rcpp::checkUserInterrupt();
  }
  return(result);
}

/* 
  log-sum-exp function
*/ 


double logsumexp(vec v){
  double mx = max(v);
  return(mx + log(sum(exp(v - mx))));
}

/* 
  Univariate Gaussian probability density function on a log-scale
*/ 

double mlog_normpdf(double x,
                    double mu,
                    double sigma){
  double z = (x - mu)/sigma;
  double out = - 0.5 * log(2.0 * M_PI * pow(sigma,2)) - 0.5 * pow(z, 2.0);
  return(out);
}

/* 
  Univariate Gaussian probability density function on a log-scale (vector version)
*/ 

vec mlog_normpdfv(vec x,
                  double mu,
                  double sigma){
  double s2 = pow(sigma,2);
  vec out(x.n_elem);
  for(uword i = 0; i < x.n_elem; i++){
    double z = (x(i) - mu)/sigma;
    out(i) = - 0.5 * log(2.0 * M_PI * s2) - 0.5 * pow(z, 2.0);
  }
  return(out);
}

/* 
  Sampling function from a discrete distribution with log-probabilities 
*/ 

int rint_log(vec lweights){
  
  double u = randu();
  vec probs(lweights.n_elem);
  for(uword k = 0; k < probs.n_elem; k++) {
    probs(k) = 1 / sum(exp(lweights - lweights(k)));
  }
  
  probs = probs / sum(probs);
  probs = cumsum(probs);
  for(uword k = 0; k < probs.n_elem; k++) {
    if(u <= probs[k]) {
      return k;
    }
  }
  return -1;
}

/* 
  Cleaning the matrix of parameters and a partition vector
    by discarding empty clusters and the corresponding parameters
*/ 

void para_clean_long(mat &group_param,
                     vec &clust) {
  int k = group_param.n_rows;
  int u_bound;

  // for all the used parameters
  for(uword i = 0; i < k; i++){

    // if a cluster is empty
    if((int) sum(clust == i) == 0){

      // find the last full cluster, then swap
      for(uword j = k; j > i; j--){
        if((int) sum(clust == j) != 0){

          // SWAPPING!!
          clust(find(clust == j) ).fill(i);
          group_param.swap_rows(i,j);
          break;
        }
      }
    }
  }

  // reduce dimensions
  u_bound = 0;
  for(arma::uword i = 0; i < k; i++){
    if(arma::accu(clust == i) > 0){
      u_bound += 1;
    }
  }

  // resize object to the correct dimension
  group_param.resize(u_bound, group_param.n_cols);
}

/* 
  Evaluate the Mahalanobis distance while including a new
    observation in a cluster (old and new distance)
*/ 

void cont_dist(mat X,
               vec clust, 
               uword j,
               vec new_val,
               vec &old_dist,
               vec &new_dist,
               mat inv_cov_centroid){
  
  mat X_temp, X_cen, X_cen2;
  rowvec coef_mean, coef_mean2;
  int nj = accu(clust == j);
  old_dist.resize(nj);
  new_dist.resize(nj);
  
  X_temp = X.rows(find(clust == j));
  X_cen.copy_size(X_temp);
  X_cen2.copy_size(X_temp);

  coef_mean = mean(X_temp, 0);
  coef_mean2 = (coef_mean * nj + new_val.t()) / (nj + 1); 
  
  for(uword r = 0; r < nj; r++){
    X_cen.row(r) = X_temp.row(r) - coef_mean;
    X_cen2.row(r) = X_temp.row(r) - coef_mean2;
  }

  old_dist = sum((X_cen * inv_cov_centroid) % X_cen, 1);
  new_dist = sum((X_cen2 * inv_cov_centroid) % X_cen2, 1);
  new_dist.resize(nj + 1);
  
  new_dist(nj) = as_scalar((new_val.t() - coef_mean2) *
    inv_cov_centroid * (new_val.t() - coef_mean2).t());
  
}

/* 
  Evaluate the Hamming distance while including a new
    observation in a cluster (old and new distance)
*/ 

void hamming_dist(mat X_disc,
                  vec clust, 
                  uword j,
                  vec new_val,
                  vec &old_dist_disc,
                  vec &new_dist_disc){
  
  double k = (double) X_disc.n_cols; 
  mat X_temp;
  int nj = accu(clust == j);
  old_dist_disc.resize(nj);
  new_dist_disc.resize(nj);
  
  X_temp = X_disc.rows(find(clust == j));
  old_dist_disc.fill(0.0);
  new_dist_disc.fill(0.0);
  
  // means
  rowvec coef_mean, coef_mean2;
  coef_mean = round(mean(X_temp, 0));
  coef_mean2 = round((coef_mean * X_temp.n_rows + new_val.t()) / (X_temp.n_rows + 1)); 
  
  for(uword l = 0; l < k; l++){
    old_dist_disc.elem(find(X_temp.col(l) != coef_mean(l))) += 1.0 / k;
    new_dist_disc.elem(find(X_temp.col(l) != coef_mean2(l))) += 1.0 / k;
  }
  
  new_dist_disc.resize(new_dist_disc.n_elem + 1);
  for(uword l = 0; l < k; l++){
    if(new_val(l) != coef_mean2(l)){
      new_dist_disc(new_dist_disc.n_elem - 1) += 1.0 / k;
    }
  }
}

/*
  Update the observation-time specific parameters (eta)
*/

void update_eta(vec clust,
                mat Y,
                vec n_is,
                mat X_fixed,
                mat X_random,
                vec coef_fixed,
                vec coef_random,
                mat group_param,
                double lambda0,
                int n_importance,
                mat &eta){
  
  double temp_mean, temp_var; 
  int temp_index;
  double tc, temp;
  vec u, temp_probs, temp_val;
  
  for(uword i = 0; i < Y.n_rows; i++){
    
    temp_index = clust(i);
    for(uword t = 0; t < n_is(i) + 1; t++){
      
      u = randu(n_importance);
      temp_val = - log(u) / lambda0;
      
      tc = X_random(i,t);
  
      temp_mean = (group_param(temp_index, 1)) / 
        (pow(group_param(temp_index, 1), 2) + group_param(temp_index, 2)) * 
        (Y(i,t) - (group_param(temp_index, 0) + 
        dot(coef_fixed, X_fixed.row(i)) + coef_random(t) * tc));
  
      temp_var = (group_param(temp_index, 2)) / (pow(group_param(temp_index, 1), 2) + group_param(temp_index, 2));
      
      temp_probs = mlog_normpdfv(temp_val, temp_mean, sqrt(temp_var)) - 
        log(lambda0) + lambda0 * temp_val;
      int idx = rint_log(temp_probs);
      eta(i,t) = temp_val(idx);
      
    }
  }
}

/*
  Sample the censored time from a truncated normal distribution
*/

void update_cens_time(vec clust,
                      mat &Y,
                      vec cens,
                      vec n_is,
                      mat X_fixed,
                      mat X_random,
                      vec coef_fixed,
                      vec coef_random,
                      mat group_param,
                      mat eta,
                      double lambda0,
                      int n_importance){
  
  vec Y_temp, u, temp_val, temp_probs;
  double t_mean, y_ini1;
  double tc;
  
  // Rcpp::Rcout << "\n\n-----------------\n\n";
  for(uword i = 0; i < Y.n_rows; i++){
    Y_temp = Y(i, span(0, n_is(i) - 1)).t();
    y_ini1 = log(cens(i) - accu(exp(Y_temp)));
    u = randu(n_importance);
    temp_val = - log(u) / lambda0 + y_ini1;
    
    tc = X_random(i,n_is(i));
    t_mean = group_param(clust(i),0) + dot(X_fixed.row(i), coef_fixed) + 
      coef_random(n_is(i) - 1) * tc + eta(i, n_is(i) - 1) * group_param(clust(i), 1);
    
    temp_probs = mlog_normpdfv(temp_val, t_mean, sqrt(group_param(clust(i), 2))) - 
      log(lambda0) + lambda0 * temp_val;
    int idx = rint_log(temp_probs);
    
    Y(i, n_is(i)) = temp_val(idx);
    
    // Rcpp::Rcout << "\n" << Y(i, n_is(i)) << "\n";
  }
}


/*
  Update the censored values of the covariates (for predictive inference)
*/

void update_cens_covs(mat Y,
                      vec n_is,
                      mat &X_random){
  
  for(uword i = 0; i < Y.n_rows; i++){
    X_random(i, n_is(i)) = X_random(i, n_is(i) - 1) + exp(Y(i, n_is(i))) / 365.25;
  }
}

/*
  Update the regression coefficients (fixed effects)
*/

void update_beta_0(vec clust,
                   mat Y,
                   vec n_is,
                   mat X_fixed,
                   mat X_random,
                   vec &coef_fixed,
                   vec coef_random,
                   mat group_param,
                   mat eta,
                   mat Sigma0){
  vec tvec, b0star;
  mat tmat, Sigma0star;
  tmat.fill(0.0);
  tvec.fill(0.0);
  int tindx = clust(0);
  double tc;
  tc = X_random(0,0);
    
  tmat = (n_is(0) + 1) * X_fixed.row(0).t() * X_fixed.row(0) / group_param(tindx,2);
  tvec = ((Y(0,0) - (group_param(tindx,0) + coef_random(0) * tc +
    group_param(tindx,1) * eta(0,0))) / group_param(tindx,2)) * X_fixed.row(0).t();
  
  for(uword t = 1; t < n_is(0) + 1; t++){
    tc = X_random(0, t);
    
    tvec += ((Y(0,t) - (group_param(tindx,0) + coef_random(t) * tc +
      group_param(tindx,1) * eta(0,t))) / group_param(tindx,2)) * X_fixed.row(0).t();
  }
  
  for(uword i = 1; i < clust.n_elem; i++){
    tindx = clust(i);
    tmat += (n_is(i) + 1) * X_fixed.row(i).t() * X_fixed.row(i) / group_param(tindx,2);
    for(uword t = 0; t < n_is(i) + 1; t++){
      tc = X_random(i, t);
      tvec += ((Y(i,t) - (group_param(tindx,0) + coef_random(t) * tc +
        group_param(tindx,1) * eta(i,t))) / group_param(tindx,2)) * X_fixed.row(i).t(); 
    }
  }
  
  Sigma0star = inv(inv(Sigma0) + tmat);
  b0star = Sigma0star * tvec;
  coef_fixed = mvnrnd(b0star, Sigma0star);
}

/*
  Update the regression coefficients (time-dependent effects)
*/

void update_beta_random(vec clust,
                        mat Y,
                        vec n_is,
                        mat X_fixed,
                        mat X_random,
                        vec coef_fixed,
                        vec &coef_random,
                        mat group_param,
                        mat eta,
                        double tau2){
  
  double K_star, accu_var, beta_star, accu_exp, tc;
  
  for(uword t = 0; t < coef_random.n_elem; t++){
    
    accu_var = 0.0;
    accu_exp = 0.0;
    for(uword i = 0; i < clust.n_elem; i++){
      
      if(n_is(i) > t){
        tc = X_random(i, t);
        
        accu_var += tc * tc / group_param(clust(i),2);
        accu_exp += ((Y(i,t) - (group_param(clust(i),0)) + dot(X_fixed.row(i), coef_fixed) + 
          group_param(clust(i),1) * eta(i,t))) / group_param(clust(i),2) * tc;
      }
    }
    
    
    K_star = 1/(1 / tau2 + accu_var);
    beta_star = K_star * (accu_exp);
    coef_random.row(t) = randn() * sqrt(K_star) + beta_star;
  }
}

/*
  Update tau
*/

void update_tau2(mat coef_random,
                 double tau2,
                 double nu0,
                 double tau0){
  
  double a_star, b_star;
  int J = coef_random.n_elem;
  a_star = nu0 + J / 2.0;
  b_star = tau0 + 0.5 * accu(pow(coef_random, 2));
  tau2 = 1.0 / randg(distr_param(a_star, 1 / b_star));  
}

/*
  Update the group-specific parameters
*/

void update_group_param(vec clust,
                        mat Y,
                        vec n_is,
                        mat X_fixed,
                        mat X_random,
                        vec coef_fixed,
                        vec coef_random,
                        mat &group_param,
                        mat eta,
                        mat k0k1,
                        vec theta_0,
                        double a0,
                        double b0){
  
  double nj, a_star, b_star, accu_square, temp, tc;
  mat accu_mat(2,2), K_star(2,2);
  vec accu_gamma(2), tvec(2), theta_star(2), tsample(2);
  tvec.fill(1.0);
  accu_gamma.fill(0.0);
  accu_square = 0.0;
  accu_mat.fill(0.0);
  
  for(uword j = 0; j < group_param.n_rows; j++){
    nj = accu(clust == j);  
    tvec.fill(1.0);
    accu_gamma.fill(0.0);
    accu_square = 0.0;
    accu_mat.fill(0.0);
    
    for(uword i = 0; i < Y.n_rows; i++){
      if(clust(i) == j){
        for(uword t = 0; t < n_is(i) + 1; t++){
          tc = X_random(i,t);
          temp = Y(i,t) - dot(X_fixed.row(i), coef_fixed) - coef_random(t) *  tc;
          tvec(1) = eta(i, t);
          accu_square += pow(temp, 2);
          accu_gamma += temp * tvec;
          accu_mat += tvec * tvec.t();
        } 
      }
    }
    
    K_star = inv(inv(k0k1) + accu_mat);
    theta_star = K_star * (accu_gamma + inv(k0k1) * theta_0);
    
    a_star = a0 + nj / 2.0;
    
    b_star = b0 + 0.5 * (accu_square + as_scalar(theta_0.t() * inv(k0k1) * theta_0) - 
      as_scalar(theta_star.t() * inv(K_star) * theta_star));
    
    group_param(j,2) = 1.0 / randg(distr_param(a_star, 1 / b_star));
    tsample = mvnrnd(theta_star, group_param(j,2) * K_star);
    group_param(j,0) = tsample(0);
    group_param(j,1) = tsample(1);
  }
}

/*
  Update the latent variables U
*/

void u_update_long(double &u,
                   double s2v,
                   double kappa,
                   double sigma,
                   vec clust, 
                   int acc_rate_u){
  
  double v_old = log(u);
  double v_temp = randn() * sqrt(s2v) + v_old;
  double u_temp = exp(v_temp);
  vec t_unique = unique(clust);
  double k = t_unique.n_elem;
  double n = clust.n_elem;
  
  double acc_ratio_log = std::min(0.0, n * (v_temp - v_old) + 
                                  (n - k * sigma) * log((1 + exp(v_old)) / (1 + exp(v_temp))) + 
                                  (kappa / sigma) * (pow(exp(v_old) + 1, sigma) - pow(exp(v_temp) + 1, sigma)));
  
  
  if(log(randu()) <= acc_ratio_log){
    u = u_temp;
    acc_rate_u += 1;
  }
}

/*
  Update the cluster allocation
*/

void update_cluster_long(vec &clust,
                         mat Y,
                         vec n_is,
                         mat X_fixed,
                         mat X_random,
                         mat X_temp_cont, 
                         mat X_temp_disc,   
                         mat X_g_temp_cont, 
                         mat X_g_temp_disc, 
                         vec coef_fixed,
                         vec coef_random,
                         mat &group_param,
                         int napprox,
                         mat k0k1,
                         vec theta_0,
                         double a0,
                         double b0,
                         mat eta, 
                         int g_fun, 
                         mat inv_cov_covs,
                         double lambda,
                         double power,
                         double sigma,
                         double kappa,
                         double u,
                         bool use_cont,
                         bool use_disc){
  
  double tsg2, out;
  int k, nj, tcl;
  vec tvec, probs;
  mat tmat(napprox, 3); 
  double tc;
  
  double all_cols = (double) (X_g_temp_cont.n_cols + X_g_temp_disc.n_cols);
  double wC = ((double) X_g_temp_cont.n_cols) / all_cols;
  double wD = ((double) X_g_temp_disc.n_cols) / all_cols;
  
  vec dist_old;
  vec dist_new;
  vec dist_old_disc;
  vec dist_new_disc;
  rowvec tvec_par;
  double tdist_new, tdist_old;
  
  int n = clust.n_elem;
  for(uword i = 0; i < n; i++){
    
    bool req_clean = (accu(clust == clust(i)) == 1);
    tcl = clust(i);
    clust(i) = group_param.n_rows + 1;
    
    if(req_clean){
      tvec_par = group_param.row(tcl);
      para_clean_long(group_param, clust);
    }
  
    k = group_param.n_rows;
    probs.resize(k+napprox);
    probs.fill(0);
  
    for(uword j = 0; j < k; j++) {
      nj = accu(clust == j);
      out = 0.0; 
      
      for(uword t = 0; t < n_is(i); t++){
        tc = X_random(i, t);
        out += mlog_normpdf(Y(i,t), group_param(j, 0) + 
          dot(coef_fixed, X_fixed.row(i)) + coef_random(t) * tc +
          eta(i,t) * group_param(j, 1), sqrt(group_param(j, 2)));
      }
  
      if(g_fun > 0){
        if(use_cont && use_disc){

          cont_dist(X_g_temp_cont, clust, j, X_g_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs); 
          hamming_dist(X_g_temp_disc, clust, j, X_g_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
  
          tdist_new = accu(wC * dist_new + wD * dist_new_disc);
          tdist_old = accu(wC * dist_old + wD * dist_old_disc);        

        } else if(use_cont){

          cont_dist(X_g_temp_cont, clust, j, X_g_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs); 
          tdist_new = accu(dist_new);
          tdist_old = accu(dist_old);

        } else if(use_disc){

          hamming_dist(X_g_temp_disc, clust, j, X_g_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
          tdist_new = accu(dist_new_disc);
          tdist_old = accu(dist_old_disc);

        }
      }
      
      if(g_fun == 0){

        probs(j) = out + log(nj - sigma); 

      } else if(g_fun == 1){

        probs(j) = out + log(nj - sigma) - pow(lambda * (tdist_new), power) + 
          pow(lambda * (tdist_old), power);

      } else if(g_fun == 2){

        probs(j) = out + log(nj - sigma) - power * log(1 + lambda * (tdist_new)) + 
          power * log(1 + lambda * (tdist_old));

      } else if(g_fun == 3){

        probs(j) = out + log(nj - sigma) - pow(lambda * (tdist_new), power) * log(1 + pow(lambda * (tdist_new), power))
        + pow(lambda * (tdist_old), power) * log(1 + pow(lambda * (tdist_old), power));

      }
    }
    
    for(uword j = k; j < k + napprox; j++) {
      
      out = 0.0; 
      tsg2 = 1.0 / randg(distr_param(a0, 1 / b0));
      tvec = mvnrnd(theta_0, k0k1);
      tmat(j - k, 0) = tvec(0);
      tmat(j - k, 1) = tvec(1);
      tmat(j - k, 2) = tsg2;
      
      if(req_clean){
        tmat.row(0) = tvec_par;
      }
      
      for(uword t = 0; t < n_is(i); t++){
        out += mlog_normpdf(Y(i,t), tvec(0) + 
          dot(coef_fixed, X_fixed.row(i)) + coef_random(t) * X_random(i, t) +
          eta(i,t) * tvec(1), sqrt(tsg2));
      }

      probs(j) = out + log(kappa) + sigma * log(u + 1.0) - log(napprox);
    }    
    
    // Rcpp::Rcout << "\n" << probs.t() << "\n";
    
    int temp_cl = rint_log(probs);
    if(temp_cl > k - 1){
      group_param.resize(k + 1, group_param.n_cols);
      group_param.row(k) = tmat.row(temp_cl - k);
      clust(i) = k;
    } else {
      clust(i) = temp_cl;
    }
  }
}

/*
  Compute LPML
*/

rowvec compute_lLIK(vec clust,
                    mat Y,
                    vec n_is,
                    mat X_fixed,
                    mat X_random,
                    mat X_temp_cont, 
                    mat X_temp_disc,   
                    mat X_g_temp_cont, 
                    mat X_g_temp_disc, 
                    vec coef_fixed,
                    vec coef_random,
                    mat group_param,
                    mat eta, 
                    int g_fun, 
                    mat inv_cov_covs,
                    double lambda,
                    double power,
                    double sigma,
                    bool use_cont,
                    bool use_disc){
  
  rowvec LPML(clust.n_elem);
  uword idx;
  vec probs;
  double accu_probs, tlog_val, tc, tsg2, out;
  int k, nj, tcl;
  

  double all_cols = (double) (X_g_temp_cont.n_cols + X_g_temp_disc.n_cols);
  double wC = ((double) X_g_temp_cont.n_cols) / all_cols;
  double wD = ((double) X_g_temp_disc.n_cols) / all_cols;
  
  vec dist_old;
  vec dist_new;
  vec dist_old_disc;
  vec dist_new_disc;
  rowvec tvec_par;
  double tdist_new, tdist_old;

  
  for(uword i = 0; i < clust.n_elem; i++){
    
    accu_probs = 0.0;
    LPML(i) = 0.0;

    k = group_param.n_rows;
    probs.resize(k);
    probs.fill(0);
    
    for(uword j = 0; j < k; j++) {

      bool unique_clust = (accu(clust == clust(i)) == 1) & (clust(i) == j);
      nj = accu(clust == j);
      out = 0.0; 
      
      for(uword t = 0; t < n_is(i); t++){
        tc = X_random(i, t);
        out += mlog_normpdf(Y(i,t), group_param(j, 0) + 
          dot(coef_fixed, X_fixed.row(i)) + coef_random(t) * tc +
          eta(i,t) * group_param(j, 1), sqrt(group_param(j, 2)));
      }

      if(g_fun > 0){
        if(unique_clust){
          tdist_new = 0.0;
          tdist_old = 0.0;
        } else {
          if(use_cont && use_disc){
            cont_dist(X_g_temp_cont, clust, j, X_g_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs); 
            hamming_dist(X_g_temp_disc, clust, j, X_g_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
            tdist_new = accu(wC * dist_new + wD * dist_new_disc);
            tdist_old = accu(wC * dist_old + wD * dist_old_disc);
          } else if(use_cont){
            cont_dist(X_g_temp_cont, clust, j, X_g_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs); 
            tdist_new = accu(dist_new);
            tdist_old = accu(dist_old);
          } else if(use_disc){
            hamming_dist(X_g_temp_disc, clust, j, X_g_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
            tdist_new = accu(dist_new_disc);
            tdist_old = accu(dist_old_disc);
          }
        }
      }
      
      if(g_fun == 0){
        tlog_val = log(nj - sigma);
      } else if(g_fun == 1){
        tlog_val = log(nj - sigma) - pow(lambda * (tdist_new), power) + 
          pow(lambda * (tdist_old), power);
      } else if(g_fun == 2){
        tlog_val = log(nj - sigma) - power * log(1 + lambda * (tdist_new)) + 
          power * log(1 + lambda * (tdist_old));
      } else if(g_fun == 3){
        tlog_val = log(nj - sigma) - pow(lambda * (tdist_new), power) * log(1 + pow(lambda * (tdist_new), power))
        + pow(lambda * (tdist_old), power) * log(1 + pow(lambda * (tdist_old), power));
      }
      probs(j) = out + tlog_val; 
      accu_probs += exp(tlog_val);
    }
    LPML(i) = log(sum(exp(probs) / accu_probs));
  }
  return(LPML);
}

/*
  Predictive function
*/

mat pred_long_full(vec clust,
                   mat Y,
                   vec n_is_new,
                   mat X_g_temp_cont, 
                   mat X_g_temp_disc, 
                   vec coef_fixed,
                   vec coef_random,
                   mat group_param,
                   mat X_new_fixed,
                   mat X_new_random,
                   mat X_new_g_temp_cont, 
                   mat X_new_g_temp_disc, 
                   double a0, 
                   double b0, 
                   vec theta_0, 
                   mat k0k1,
                   int napprox,
                   int g_fun,
                   double sigma,
                   mat inv_cov_covs,
                   double lambda,
                   double power,
                   double kappa,
                   double u, 
                   bool use_cont, 
                   bool use_disc){
  
  mat results(X_new_fixed.n_rows, Y.n_cols);
  vec probs(group_param.n_rows + 1);
  
  results.fill(0.0);
  probs.fill(0.0);
  
  vec temp_eta(Y.n_cols);
  int temp_cl, nj;
  int temp_col = X_new_g_temp_cont.n_cols + X_new_g_temp_disc.n_cols;
  double wC = ((double) X_new_g_temp_cont.n_cols) / ((double) temp_col);
  double wD = ((double) X_new_g_temp_disc.n_cols) / ((double) temp_col);

  vec dist_old, dist_new, dist_old_disc, dist_new_disc, tvec(3), tvec2(2);
  double tdist_new, tdist_old;
  
  for(uword i = 0; i < X_new_fixed.n_rows; i ++){
    for(uword r = 0; r < n_is_new(i); r++){
      temp_eta(r) = abs(randn());
    }
    for(uword j = 0; j < group_param.n_rows; j ++){
      if(use_cont && use_disc){
        cont_dist(X_g_temp_cont, clust, j, X_new_g_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs); 
        hamming_dist(X_g_temp_disc, clust, j, X_new_g_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
        tdist_new = accu(wC * dist_new + wD * dist_new_disc);
        tdist_old = accu(wC * dist_old + wD * dist_old_disc);
        
      } else if(use_cont){
        cont_dist(X_g_temp_cont, clust, j, X_new_g_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs); 
        tdist_new = accu(dist_new);
        tdist_old = accu(dist_old);
      } else if(use_disc){
        hamming_dist(X_g_temp_disc, clust, j, X_new_g_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
        tdist_new = accu(dist_new_disc);
        tdist_old = accu(dist_old_disc);
      }
      nj = accu(clust == j);

      if(g_fun == 0){
        probs(j) = (nj - sigma);
      } else if(g_fun == 1){
        probs(j) = (nj - sigma) * exp(-pow(lambda * (tdist_new), power) + pow(lambda * (tdist_old), power));
      } else if(g_fun == 2){
        probs(j) = (nj - sigma) * exp(-power * log(1 + lambda * (tdist_new)) + power * log(1 + lambda * (tdist_old)));
      } else if(g_fun == 3){
        probs(j) = (nj - sigma) * exp(-pow(lambda * (tdist_new), power) * log(1 + pow(lambda * (tdist_new), power))
        + pow(lambda * (tdist_old), power) * log(1 + pow(lambda * (tdist_old), power)));
      }
      
      for(uword r = 0; r < n_is_new(i); r++){
        results(i,r) += probs(j) * (group_param(j, 0) + 
          dot(coef_fixed, X_new_fixed.row(i)) + coef_random(r) * X_new_random(i,r) +
          temp_eta(r) * group_param(j, 1));
      }
    }
    
    probs(group_param.n_rows) = (kappa) * pow(u + 1.0, sigma);
    double aw = accu(probs);

    for(uword j = 0; j < napprox; j++) {
      tvec(2) = 1.0 / randg(distr_param(a0, 1 / b0));
      tvec2 = mvnrnd(theta_0, k0k1);
      tvec(0) = tvec2(0);
      tvec(1) = tvec2(1);

      for(uword r = 0; r < n_is_new(i); r++){
        results(i,r) += probs(group_param.n_rows) / napprox * (tvec(0) + 
          dot(coef_fixed, X_new_fixed.row(i)) + coef_random(r) * X_new_random(i,r) +
          temp_eta(r) * tvec(1));
      }
    }
    
    for(uword r = 0; r < n_is_new(i); r++){
      results(i,r) /=  aw;
    }
  }
  return results;
}


/*
  MAIN FUNCTION
*/

// [[Rcpp::export]]
Rcpp::List main_long(mat YY,
                     vec cens,
                     mat X_fixed,
                     mat X_random,
                     mat X_g,
                     vec vartype,
                     vec vartype_g,
                     vec n_is,
                     int niter,
                     int nburn,
                     int thin,
                     mat k0k1, 
                     vec theta_0,
                     double a0, 
                     double b0,
                     double kappa,
                     double sigma,
                     double s2v,
                     double nu0, 
                     double tau0,
                     int g_fun,
                     double lambda,
                     double power,
                     double lambda0, 
                     int n_importance,
                     int napprox,
                     mat Sigma0, 
                     int nupd,
                     mat X_new_fixed, 
                     mat X_new_random, 
                     mat X_new_g,
                     vec n_is_new, 
                     mat inv_cov_covs,
                     bool prevision_full = false, 
                     bool prevision_last = false){
  
  bool use_cont = false;
  bool use_disc = false;
  
  if(accu(vartype_g == 1) > 1){
    use_cont = true;
  }
  if(accu(vartype_g == 0) > 1){
    use_disc = true;
  }
  
  mat X_temp_cont = X_fixed.cols(find(vartype == 1));
  mat X_temp_disc = X_fixed.cols(find(vartype == 0));
  
  mat X_g_temp_cont = X_g.cols(find(vartype_g == 1));
  mat X_g_temp_disc = X_g.cols(find(vartype_g == 0));
  
  mat X_new_g_temp_cont;
  mat X_new_g_temp_disc;
  
  if(prevision_full == true){
    X_new_g_temp_cont = X_new_g.cols(find(vartype_g == 1));
    X_new_g_temp_disc = X_new_g.cols(find(vartype_g == 0));
  }
  
  mat Y = YY;
  Y.resize(Y.n_rows, Y.n_cols + 1);
  
  // output
  int nitem = (niter - nburn) / thin;
  field<vec> res_beta_fixed(nitem);
  field<vec> res_beta_random(nitem);
  field<mat> res_group_param(nitem);
  field<mat> pred_full(nitem);
  mat res_clust(nitem, Y.n_rows);
  vec u_result(nitem);
  vec s2v_result(nitem);
  mat LPML_out(nitem, Y.n_rows);
  
  //quantities
  vec clust(Y.n_rows);
  mat eta(Y.n_rows, Y.n_cols + 1);
  double tau2;
  
  clust.fill(0);
  eta.fill(1.0);
  tau2 = 1.0;
  
  vec coef_fixed(X_fixed.n_cols);
  vec coef_random(X_random.n_cols + 1);
  mat group_param(1, 3);
  
  coef_fixed.fill(0.0);
  coef_random.fill(0.0);
  group_param.fill(0.0);
  group_param(0,2) = 1.0;
  
  int acc_rate_u = 0;
  double u = 1;
  int res_index = 0;
  
  // main loop
  int start_s = clock();
  int current_s;
  for(uword iter = 0; iter < niter; iter++){
    
    update_cens_time(clust, Y, cens, n_is, X_fixed, X_random, coef_fixed,
                     coef_random, group_param, eta, lambda0, n_importance);
    
    para_clean_long(group_param, clust);
    
    update_eta(clust, Y, n_is, X_fixed, X_random, coef_fixed, coef_random, group_param, lambda0, n_importance, eta);
    
    update_beta_0(clust, Y, n_is, X_fixed, X_random, coef_fixed, coef_random, group_param, eta, Sigma0);
    
    update_beta_random(clust, Y, n_is, X_fixed, X_random, coef_fixed, coef_random, group_param, eta, tau2);
    
    update_tau2(coef_random, tau2, nu0, tau0);
    
    u_update_long(u, s2v, kappa, sigma, clust, acc_rate_u);
    
    update_group_param(clust, Y, n_is, X_fixed, X_random, coef_fixed, coef_random,
                       group_param, eta, k0k1, theta_0, a0, b0);
    
    update_cluster_long(clust, Y, n_is, X_fixed, X_random, X_temp_cont, X_temp_disc, 
                        X_g_temp_cont, X_g_temp_disc, coef_fixed,
                        coef_random, group_param, napprox, k0k1, theta_0, a0, b0, eta, g_fun, 
                        inv_cov_covs, lambda, power, sigma, kappa, u, use_cont, use_disc);
    
    // save the results
    if((iter >= nburn) & ((iter + 1) % thin == 0)){
      res_clust.row(res_index) = clust.t();
      u_result(res_index) = u;
      res_beta_fixed(res_index) = coef_fixed;
      res_beta_random(res_index) = coef_random;
      res_group_param(res_index) = group_param;
      s2v_result(res_index) = s2v;
      
      LPML_out.row(res_index) = compute_lLIK(clust, Y, n_is, X_fixed, X_random,
                         X_temp_cont, X_temp_disc, X_g_temp_cont, X_g_temp_disc, 
                         coef_fixed, coef_random, group_param, eta, g_fun, 
                         inv_cov_covs, lambda, power, sigma, use_cont, use_disc);
      
      if(prevision_full == true){
        pred_full(res_index) = pred_long_full(clust, Y, n_is_new, X_g_temp_cont, X_g_temp_disc,
                  coef_fixed, coef_random, group_param, X_new_fixed,
                  X_new_random, X_new_g_temp_cont, X_new_g_temp_disc,
                  a0, b0, theta_0, k0k1, napprox, g_fun, sigma, inv_cov_covs, lambda,
                  power, kappa, u, use_cont, use_disc);
      }
      res_index += 1;
    }
    
    if((iter + 1) % nupd == 0){
      current_s = clock();
      Rcpp::Rcout << "Completed:\t" << (iter + 1) << "/" << niter << " - in " <<
        double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
    }
    Rcpp::checkUserInterrupt();
  }
  int time = clock() - start_s;
  
  Rcpp::List results;
  results["time"] = time/CLOCKS_PER_SEC;
  results["k0k1"] = k0k1;
  results["theta_0"] = theta_0;
  results["a0"] = a0;
  results["b0"] = b0;
  results["kappa"] = kappa;
  results["sigma"] = sigma;
  results["u"] = u_result;
  results["res_beta_fixed"] = res_beta_fixed;
  results["res_beta_random"] = res_beta_random;
  results["res_group_param"] = res_group_param;
  results["clust"] = res_clust;
  results["LPML"] = LPML_out;
  results["pred_full"] = pred_full;
  return results;
}