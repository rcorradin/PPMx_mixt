// #define ARMA_DONT_PRINT_ERRORS
#include "RcppArmadillo.h"
#include "RcppDist.h"
using namespace arma;
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

// density function of a location-scale t-student distribution
double dt_ls(double x,
             double df,
             double mu,
             double sigma){
  double z = (x - mu)/sigma;
  double out = lgamma((df + 1) / 2) - log(sqrt(M_PI * df)) - log(sigma) -
    lgamma(df / 2) - (df + 1) * std::log1p(z * z / df) / 2;
  return(out);
}

// sample on a logarithmic scale
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

// ----------------- COVS DIST --------------------------------------------------------

// [[Rcpp::export]]
void cont_dist(mat X,
               vec clust, 
               uword j,
               vec new_val,
               vec &old_dist,
               vec &new_dist,
               mat inv_cov_centroid){
  mat X_temp;
  rowvec coef_mean, coef_mean2;
  int nj = accu(clust == j);
  old_dist.resize(nj);
  new_dist.resize(nj + 1);
  
  X_temp = X.rows(find(clust == j));
  coef_mean = mean(X_temp, 0);
  coef_mean2 = (coef_mean * nj + new_val.t()) / (nj + 1); 
  
  for(uword r = 0; r < nj; r++){
    old_dist(r) = as_scalar((X_temp.row(r) - coef_mean) *
      inv_cov_centroid * (X_temp.row(r) - coef_mean).t());
    
    new_dist(r) = as_scalar((X_temp.row(r) - coef_mean2) *
      inv_cov_centroid * (X_temp.row(r) - coef_mean2).t());
  }
  
  new_dist(new_dist.n_elem - 1) = as_scalar((new_val.t() - coef_mean2) *
    inv_cov_centroid * (new_val.t() - coef_mean2).t());
}

// [[Rcpp::export]]
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
  new_dist_disc.resize(nj + 1);
  X_temp = X_disc.rows(find(clust == j));
  old_dist_disc.fill(0.0);
  new_dist_disc.fill(0.0);
  
  // means
  rowvec coef_mean, coef_mean2;
  coef_mean = round(mean(X_temp, 0));
  coef_mean2 = round((coef_mean * X_temp.n_rows + new_val.t()) / (X_temp.n_rows + 1)); 
  
  for(uword r = 0; r < nj; r++){
    for(uword l = 0; l < k; l++){
      if(X_temp(r,l) != coef_mean(l)){
        old_dist_disc(r) += 1.0 / k;
      }
      
      if(X_temp(r,l) != coef_mean2(l)){
        new_dist_disc(r) += 1.0 / k;
      }
    }
  }
  
  for(uword l = 0; l < k; l++){
    if(new_val(l) != coef_mean(l)){
      new_dist_disc(new_dist_disc.n_elem - 1) += 1.0 / k;
    }
  }
}

//--------------------------------------
//--------------------------------------
// Gaussian kernel - regression
//--------------------------------------
//--------------------------------------

// clean the parameters
void para_clean(mat &beta,
                vec &sigma2,
                vec &clust) {
  int k = beta.n_rows;
  int u_bound;
  
  // for all the used parameters
  for(uword i = 0; i < k; i++){
    
    // if a cluster is empty
    if((int) sum(clust == i) == 0){
      
      // find the last full cluster, then swap
      for(uword j = k; j > i; j--){
        if((int) sum(clust == j) != 0){
          
          // SWAPPING!!
          clust(find(clust == j)).fill(i);
          sigma2.swap_rows(i,j);
          beta.swap_rows(i,j);
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
  beta.resize(u_bound, beta.n_cols);
  sigma2.resize(u_bound);
}

// UPDATE THETA (acceleration step)

// [[Rcpp::export]]
void update_theta_reg(vec y,
                      mat X,
                      vec clust,
                      mat &beta,
                      vec &sigma2,
                      mat B0,
                      vec m0,
                      double a0, 
                      double b0,
                      int niter, 
                      int iter){
  mat sub_X;
  vec sub_y;
  
  mat B_star;
  vec mu_star;
  double a_star;
  double b_star;
  
  for(uword j = 0; j < sigma2.n_elem; j++){
    sub_X = X.rows(find(clust == j));
    sub_y = y.elem(find(clust == j));
    
    B_star = inv(inv(B0) + sub_X.t() * sub_X);
    mu_star = B_star * (inv(B0) * m0 + sub_X.t() * sub_y);
    
    a_star = a0 + 0.5 * sub_y.n_elem;
    b_star = b0 + 0.5 * (as_scalar(m0.t() * inv(B0) * m0) + accu(pow(sub_y, 2)) -
      as_scalar(mu_star.t() * inv(B_star) * mu_star));
    
    sigma2(j) = 1.0 / randg(distr_param(a_star, 1 / b_star));
    beta.row(j) = mvnrnd(mu_star, sigma2(j) * B_star).t();
  }
}

//--------------------------------------
// UPDATE U (MH step)

// [[Rcpp::export]]
void u_update(double &u,
              double s2v,
              double kappa,
              double sigma,
              vec clust,
              int &acc_rate_u){
  
  // gaussian MH on log-scale 
  // in the spirit of FT (2013)
  
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

//--------------------------------------
// UPDATE the clusters

// [[Rcpp::export]]
void update_clust_reg(vec y,
                      mat X,
                      mat X_temp_cont, 
                      mat X_temp_disc,
                      vec &clust,
                      mat &beta,
                      vec &sigma2,
                      mat B0,
                      vec m0,
                      double a0, 
                      double b0,
                      double kappa,
                      double sigma,
                      double u,
                      int g_fun,
                      double lambda,
                      double power,
                      mat inv_cov_covs){
  vec tclust;
  vec probs;
  vec mu_star;
  rowvec coef_mean;
    
  mat B_star;
  mat X_temp;
  
  int n = clust.n_elem;
  int nj = 0;
  int k;
  double wC = ((double) X_temp_cont.n_cols) / ((double) X.n_cols);
  double wD = ((double) X_temp_disc.n_cols) / ((double) X.n_cols);
  
  double out;
  double t_sig;
  double t_mu; 
  double a_star;
  double b_star;
  
  vec dist_old;
  vec dist_new;
  vec dist_old_disc;
  vec dist_new_disc;
  double tdist_new, tdist_old;
  
  
  for(uword i = 0; i < n; i++){
    
    // clean if needed
    bool req_clean = (accu(clust == clust(i)) == 1);
    clust(i) = beta.n_rows + 1;
    if(req_clean){
      para_clean(beta, sigma2, clust);
    }
    
    // initialize useful quantities
    k = beta.n_rows;
    probs.resize(k+1);
    probs.fill(0);
    
    // compute probabilities vector
    for(uword j = 0; j < k; j++) {
      nj = accu(clust == j);
      out = log_normpdf(y(i), dot(X.row(i), beta.row(j)), sqrt(sigma2(j)));
        
        // - 0.5 * log(2 * M_PI * sigma2(j)) - 
        // 0.5 * pow(y(i) - dot(X.row(i), beta.row(j)), 2) / sigma2(j);
      
      // if g > 0, compute the centroid and the 
      // distances with and without the i-th element included
      if(g_fun > 0){
        cont_dist(X_temp_cont, clust, j, X_temp_cont.row(i).t(), dist_old, dist_new, inv_cov_covs);
        hamming_dist(X_temp_disc, clust, j, X_temp_disc.row(i).t(), dist_old_disc, dist_new_disc);
        
        // tdist_new = (accu(wC * dist_new + wD * dist_new_disc));
        // tdist_old = (accu(wC * dist_old + wD * dist_old_disc));
        tdist_new = accu(wC * dist_new + wD * dist_new_disc);
        tdist_old = accu(wC * dist_old + wD * dist_old_disc);
      }
      
      // Rcpp::Rcout << "\nnew\n" << dist_new.t() << "\n\n--------------\nold\n" << dist_old.t() << "\n\n--------------\n\n";
      // Rcpp::Rcout <<  "\n\n--------------\nDIFF\n" << wC << "\t\t" << wD << "\n\n--------------\n\n";
      
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
    
    // predictive is a t-Stud
    t_sig = as_scalar(X.row(i) * B0 * X.row(i).t());
    t_mu  = as_scalar(X.row(i) * m0);
    probs(k) = log(kappa) + sigma * log(u + 1.0) + dt_ls(y(i), 2.0 * a0, t_mu, sqrt(t_sig));
    
    // sample new
    
    int temp_cl = rint_log(probs);
    clust(i) = temp_cl;
    
    if(temp_cl == k){
      beta.resize(k+1, beta.n_cols);
      sigma2.resize(k+1);
      
      B_star = inv(inv(B0) + X.row(i).t() * X.row(i));
      mu_star = B_star * (inv(B0) * m0 + X.row(i).t() * y(i));
      a_star = a0 + (1.0 / 2.0);
      b_star = b0 + (as_scalar(m0.t() * inv(B0) * m0) + pow(y(i), 2) -
        as_scalar(mu_star.t() * inv(B_star) * mu_star)) / 2.0;
      
      sigma2(k) = 1.0 / randg(distr_param(a_star, 1 / b_star));
      beta.row(k) = mvnrnd(mu_star, sigma2(k) * B_star).t();
      
    }
  }
}

//--------------------------------------
// prediction

vec pred_values(mat Xnew, 
                mat Xnew_cont,
                mat Xnew_disc,
                mat X_temp_cont, 
                mat X_temp_disc,
                vec clust,
                mat beta,
                vec sigma2,
                int g_fun,
                double sigma,
                mat inv_cov_covs,
                double lambda,
                double power,
                double kappa,
                double u,
                mat B0,
                vec m0,
                double a0, 
                double b0){
  
  vec results(Xnew.n_rows);
  vec weights(beta.n_rows + 1);
  vec temp_probs(beta.n_rows + 1);
  
  results.fill(0.0);
  weights.fill(0.0);
  temp_probs.fill(0.0);
  int temp_cl;
  double wC = ((double) X_temp_cont.n_cols) / ((double) Xnew.n_cols - 1.0);
  double wD = ((double) X_temp_disc.n_cols) / ((double) Xnew.n_cols - 1.0);
  
  vec dist_old;
  vec dist_new;
  vec dist_old_disc;
  vec dist_new_disc;
  double tdist_new, tdist_old;
  double t_sig;
  double t_mu;
  
  for(uword i = 0; i < Xnew.n_rows; i ++){
    for(uword j = 0; j < beta.n_rows; j ++){
      
      if(g_fun > 0){
        cont_dist(X_temp_cont, clust, j, Xnew_cont.row(i).t(), dist_old, dist_new, inv_cov_covs);
        hamming_dist(X_temp_disc, clust, j, Xnew_disc.row(i).t(), dist_old_disc, dist_new_disc);
        
        // tdist_new = accu((wC * dist_new + wD * dist_new_disc));
        // tdist_old = accu((wC * dist_old + wD * dist_old_disc));
        tdist_new = accu(wC * dist_new + wD * dist_new_disc);
        tdist_old = accu(wC * dist_old + wD * dist_old_disc);
      }
        
      // Rcpp::Rcout << "\n\n" << dist_old << "\n\n" << dist_new << "\n\n" << dist_old_disc << "\n\n" << dist_new_disc << "\n\n";
      
      if(g_fun == 0){
        weights(j) = log(accu(clust == j) - sigma);
      } else if(g_fun == 1){
        // CONTROLLA I SEGNI DELLE POTENZE
        weights(j) = log(accu(clust == j) - sigma) - pow(lambda * (tdist_new), power) + 
          pow(lambda * (tdist_old), power);
      } else if(g_fun == 2){
        weights(j) = log(accu(clust == j) - sigma) - power * log(1 + lambda * (tdist_new)) + 
          power * log(1 + lambda * (tdist_old));
      } else if(g_fun == 3){
        weights(j) = log(accu(clust == j) - sigma) - pow(lambda * (tdist_new), power) * log(1 + pow(lambda * (tdist_new), power))
          + pow(lambda * (tdist_old), power) * log(1 + pow(lambda * (tdist_old), power));
        
        // if(dist_new + dist_new_disc < 1 / exp(1.0)){
        //   weights(j) += ((1 / exp(1.0)) - 1) - log(dist_new + dist_new_disc);
        // } else {
        //   weights(j) += - (dist_new + dist_new_disc) * log(dist_new + dist_new_disc);
        // }
        // 
        // if(1 + lambda * (dist_old + dist_old_disc) < 1 / exp(1.0)){
        //   weights(j) -= ((1 / exp(1.0)) - 1) - log(lambda * (dist_old + dist_old_disc));
        // } else {
        //   weights(j) -= - lambda * (dist_old + dist_old_disc) * log(lambda * (dist_old + dist_old_disc));
        // }
      }
    } 
    weights(beta.n_rows) = log(kappa) + sigma * log(u + 1.0);
    
    // compute the probabilities and sample
    for(uword k = 0; k < weights.n_elem; k++) {
      temp_probs(k) = 1 / sum(exp(weights - weights(k)));
    }
    
    temp_cl = rint_log(temp_probs);
    
    // Rcpp::Rcout << weights << "\n\n\n" << temp_cl << "\n\n\n";
    // sample the predicted value
    if(temp_cl == weights.n_elem - 1){
      t_sig = as_scalar(Xnew.row(i) * B0 * Xnew.row(i).t());
      t_mu  = as_scalar(Xnew.row(i) * m0);
      results(i) = R::rt(2 * a0) * sqrt(t_sig) + t_mu;
    } else {
      // Rcpp::Rcout << sqrt(sigma2(temp_cl)) << "\n\n\n" << dot(beta.row(temp_cl), Xnew.row(i)) << "\n\n\n";
      results(i) = randn() * sqrt(sigma2(temp_cl)) + dot(Xnew.row(i), beta.row(temp_cl)); 
    }
  }
  return results;
}

// pred grid
vec pred_values_grid(mat Xnew, 
                     mat Xnew_cont,
                     mat Xnew_disc,
                     mat X_temp_cont, 
                     mat X_temp_disc,
                     vec clust,
                     mat beta,
                     vec sigma2,
                     int g_fun,
                     double sigma,
                     mat inv_cov_covs,
                     double lambda,
                     double power,
                     double kappa,
                     double u, 
                     mat B0, 
                     vec m0, 
                     double a0, 
                     double b0,
                     vec grid){
  
  vec results(Xnew.n_rows);
  vec weights(beta.n_rows + 1);
  vec temp_probs(beta.n_rows + 1);
  mat temp_dens(beta.n_rows + 1, grid.n_elem);
  vec tdens(grid.n_elem);
  
  results.fill(0.0);
  weights.fill(0.0);
  temp_probs.fill(0.0);
  int temp_cl;
  double wC = ((double) X_temp_cont.n_cols) / ((double) Xnew.n_cols - 1.0);
  double wD = ((double) X_temp_disc.n_cols) / ((double) Xnew.n_cols - 1.0);
  
  vec dist_old;
  vec dist_new;
  vec dist_old_disc;
  vec dist_new_disc;
  double tdist_new, tdist_old;
  double t_sig;
  double t_mu;
  
  for(uword i = 0; i < Xnew.n_rows; i ++){
    for(uword j = 0; j < beta.n_rows; j ++){
      
      if(g_fun > 0){
        cont_dist(X_temp_cont, clust, j, Xnew_cont.row(i).t(), dist_old, dist_new, inv_cov_covs);
        hamming_dist(X_temp_disc, clust, j, Xnew_disc.row(i).t(), dist_old_disc, dist_new_disc);
        
        // tdist_new = accu((wC * dist_new + wD * dist_new_disc));
        // tdist_old = accu((wC * dist_old + wD * dist_old_disc));
        tdist_new = accu(wC * dist_new + wD * dist_new_disc);
        tdist_old = accu(wC * dist_old + wD * dist_old_disc);
      }
      
      if(g_fun == 0){
        weights(j) = (accu(clust == j) - sigma);
      } else if(g_fun == 1){
        // CONTROLLA I SEGNI DELLE POTENZE
        weights(j) = (accu(clust == j) - sigma) * exp(-pow(lambda * (tdist_new), power) + 
          pow(lambda * (tdist_old), power));
      } else if(g_fun == 2){
        weights(j) = (accu(clust == j) - sigma) * exp( - power * log(1 + lambda * (tdist_new)) + 
          power * log(1 + lambda * (tdist_old)));
      } else if(g_fun == 3){
        weights(j) = (accu(clust == j) - sigma) * exp( - pow(lambda * (tdist_new), power) * log(1 + pow(lambda * (tdist_new), power))
          + pow(lambda * (tdist_old), power) * log(1 + pow(lambda * (tdist_old), power)));
        
      //   if(dist_new + dist_new_disc < 1 / exp(1.0)){
      //     weights(j) += ((1 / exp(1.0)) - 1) - log(dist_new + dist_new_disc);
      //   } else {
      //     weights(j) += - (dist_new + dist_new_disc) * log(dist_new + dist_new_disc);
      //   }
      // 
      //   if(1 + lambda * (dist_old + dist_old_disc) < 1 / exp(1.0)){
      //     weights(j) -= ((1 / exp(1.0)) - 1) - log(lambda * (dist_old + dist_old_disc));
      //   } else {
      //     weights(j) -= - lambda * (dist_old + dist_old_disc) * log(lambda * (dist_old + dist_old_disc));
      //   }
      }
      for(uword l = 0; l < grid.n_elem; l++){
        temp_dens(j,l) = weights(j) * normpdf(grid(l), dot(Xnew.row(i), beta.row(j)), sqrt(sigma2(j)));
        
        // exp(- 0.5 * log(2 * M_PI * sigma2(j)) -
        //   0.5 * pow(grid(l) - as_scalar(Xnew.row(i) * beta.row(j).t()), 2) / sigma2(j));
      }
    } 
    
    weights(beta.n_rows) = (kappa) * pow(u + 1.0, sigma);
    double aw = accu(weights);
    
    t_sig = as_scalar(Xnew.row(i) * B0 * Xnew.row(i).t());
    t_mu  = as_scalar(Xnew.row(i) * m0);
    for(uword l = 0; l < grid.n_elem; l++){
      temp_dens(beta.n_rows,l) = weights(beta.n_rows) * exp(lgamma((2.0 * a0 + 1.0) / 2.0) - 
        log(t_sig * sqrt(2.0 * a0 * M_PI)) - lgamma(a0) - 
        ((2.0 * a0 + 1.0) / 2.0) * log((2.0 * a0 + pow((grid(l) - t_mu), 2) / t_sig) / (2.0 * a0)));
      temp_dens.col(l) /= aw;
    }
    
    results(i) = dot(sum(temp_dens, 0), grid) / accu(temp_dens);
  }
  return results;
}

// pred grid
vec pred_values_grid2(mat Xnew, 
                      mat Xnew_cont,
                      mat Xnew_disc,
                      mat X_temp_cont, 
                      mat X_temp_disc,
                      vec clust,
                      mat beta,
                      vec sigma2,
                      int g_fun,
                      double sigma,
                      mat inv_cov_covs,
                      double lambda,
                      double power,
                      double kappa,
                      double u, 
                      mat B0, 
                      vec m0, 
                      double a0, 
                      double b0,
                      vec grid){
  
  vec results(Xnew.n_rows);
  vec weights(beta.n_rows + 1);
  vec temp_probs(beta.n_rows + 1);
  mat temp_dens(beta.n_rows + 1, grid.n_elem);
  vec tdens(grid.n_elem);
  
  results.fill(0.0);
  weights.fill(0.0);
  temp_probs.fill(0.0);
  int temp_cl;
  double wC = ((double) X_temp_cont.n_cols) / ((double) Xnew.n_cols - 1.0);
  double wD = ((double) X_temp_disc.n_cols) / ((double) Xnew.n_cols - 1.0);
  
  vec dist_old;
  vec dist_new;
  vec dist_old_disc;
  vec dist_new_disc;
  
  double tdist_new, tdist_old;
  double t_sig;
  double t_mu;
  
  for(uword j = 0; j < beta.n_rows; j ++){
    
    // update the distance
    if(g_fun > 0){
      cont_dist(X_temp_cont, clust, j, Xnew_cont.row(0).t(), dist_old, dist_new, inv_cov_covs);
      hamming_dist(X_temp_disc, clust, j, Xnew_disc.row(0).t(), dist_old_disc, dist_new_disc);
      
      // tdist_new = accu((wC * dist_new + wD * dist_new_disc));
      // tdist_old = accu((wC * dist_old + wD * dist_old_disc));
      tdist_new = accu(wC * dist_new + wD * dist_new_disc);
      tdist_old = accu(wC * dist_old + wD * dist_old_disc);
    }
    
    
    // compute the penalizing factor
    if(g_fun == 0){
      weights(j) = (accu(clust == j) - sigma);
    } else if(g_fun == 1){
      weights(j) = (accu(clust == j) - sigma) * exp(-pow(lambda * (tdist_new), power) + 
        pow(lambda * (tdist_old), power));
    } else if(g_fun == 2){
      weights(j) = (accu(clust == j) - sigma) * exp( - power * log(1 + lambda * (tdist_new)) + 
        power * log(1 + lambda * (tdist_old)));
    } else if(g_fun == 3){
      weights(j) = (accu(clust == j) - sigma) * exp( - pow(lambda * (tdist_new), power) * log(1 + pow(lambda * (tdist_new), power))
      + pow(lambda * (tdist_old), power) * log(1 + pow(lambda * (tdist_old), power)));
    }
    
    // evaluate on the grid
    for(uword l = 0; l < grid.n_elem; l++){
      // temp_dens(j,l) = weights(j) * exp(- 0.5 * log(2 * M_PI * sigma2(j)) -
      //   0.5 * pow(grid(l) - dot(Xnew.row(0), beta.row(j)), 2) / sigma2(j));
      temp_dens(j,l) = weights(j) * normpdf(grid(l), dot(Xnew.row(0), beta.row(j)), sqrt(sigma2(j)));
    } 
  }
  
  weights(beta.n_rows) = (kappa) * pow(u + 1.0, sigma);
  double aw = accu(weights);
  
  // Rcpp::Rcout << weights.t() / aw << "\n\n";
  
  t_sig = as_scalar(Xnew.row(0) * B0 * Xnew.row(0).t());
  t_mu  = as_scalar(Xnew.row(0) * m0);
  for(uword l = 0; l < grid.n_elem; l++){
    temp_dens(temp_dens.n_rows - 1, l) = weights(beta.n_rows) * exp(lgamma((2.0 * a0 + 1.0) / 2.0) -
      log(t_sig * sqrt(2.0 * a0 * M_PI)) - lgamma(a0) -
      ((2.0 * a0 + 1.0) / 2.0) * log((2.0 * a0 + pow((grid(l) - t_mu), 2) / t_sig) / (2.0 * a0)));
    temp_dens.col(l) /= aw;
  }
  
  
  
  // Rcpp::Rcout << temp_dens.col(170).t() << "\n\n";
  
  results = sum(temp_dens, 0).t();
  
  
  return results;
}

//--------------------------------------
// MAIN

// [[Rcpp::export]]
Rcpp::List main_reg(vec y,
                    mat X_dat,
                    vec vartype,
                    int niter,
                    int nburn,
                    int thin,
                    mat B0,
                    vec m0,
                    double a0, 
                    double b0,
                    double kappa,
                    double sigma,
                    double s2v,
                    int g_fun,
                    double lambda,
                    double power,
                    bool pred = false,
                    mat Xnew_dat = mat(1,1),
                    bool pred_grid = false, 
                    vec grid = vec(1)){
  
  //-----------------------------------------
  // divide cont and disc covs
  
  mat X_temp_cont = X_dat.cols(find(vartype == 1));
  mat X_temp_disc = X_dat.cols(find(vartype == 0));
  
  //-----------------------------------------
  
  vec intercept(X_dat.n_rows);
  intercept.fill(1.0);
  mat X = join_rows(intercept, X_dat);
  
  //-----------------------------------------
  // output
  int nitem = (niter - nburn) / thin;
  field<mat> res_beta(nitem);
  field<vec> res_sigma2(nitem);
  mat res_clust(nitem, y.n_elem);
  vec u_result(nitem);
  vec s2v_result(nitem);
  
  //-----------------------------------------
  // prediction
  mat prediction;
  mat prediction2;
  mat prediction3;
  mat Xnew_cont;
  mat Xnew_disc;
  mat Xnew;
  
  if(pred == true){
    prediction.resize(nitem, Xnew_dat.n_rows);
    if(pred_grid == true){
      prediction2.resize(nitem, Xnew_dat.n_rows);
      prediction3.resize(nitem, grid.n_elem);
    }
    Xnew_cont = Xnew_dat.cols(find(vartype == 1));
    Xnew_disc = Xnew_dat.cols(find(vartype == 0));
    vec new_intercept(Xnew_dat.n_rows);
    new_intercept.fill(1.0);
    Xnew = join_rows(new_intercept, Xnew_dat);
  }
  
  //-----------------------------------------
  //quantities
  vec clust = regspace(0,  y.n_elem - 1);
  mat beta(y.n_elem, X.n_cols);
  beta.fill(0);
  
  vec sigma2(y.n_elem);
  sigma2.fill(1);
  
  int acc_rate_u = 0;
  double alpha = 1;
  double u = 1;
  int res_index = 0;
  
  mat cov_covs = cov(X_temp_cont);
  mat inv_cov_covs = inv(cov_covs);
  
  //loop
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  for(uword iter = 0; iter < niter; iter++){
    
    // Rcpp::Rcout << "\nqui1";
    para_clean(beta, sigma2, clust);
    // Rcpp::Rcout << "\nqui2";
    update_theta_reg(y, X, clust, beta, sigma2, B0, m0, a0, b0, niter, iter);
    // Rcpp::Rcout << "\nqui3";
    u_update(u, s2v, kappa, sigma, clust, acc_rate_u);
    // Rcpp::Rcout << "\nqui4";
    update_clust_reg(y, X, X_temp_cont, X_temp_disc, clust, beta, sigma2, B0, m0, a0, b0, kappa, sigma, u, g_fun, lambda, power, inv_cov_covs);
    // Rcpp::Rcout << "\nqui5";
    // Rcpp::Rcout << "\n" << clust.t();
    
    if((iter >= nburn) & ((iter + 1) % thin == 0)){
      
      res_clust.row(res_index) = clust.t();
      u_result(res_index) = u;
      res_beta(res_index) = beta;
      res_sigma2(res_index) = sigma2;
      s2v_result(res_index) = s2v;
      
      if(pred == true){
        // prediction.row(res_index) = pred_values(Xnew, Xnew_cont, Xnew_disc, X_temp_cont, X_temp_disc,
        //                clust, beta, sigma2, g_fun, sigma, inv_cov_covs, lambda, power, kappa, u,
        //                B0, m0, a0, b0).t();
        if(pred_grid == true){
          prediction2.row(res_index) = pred_values_grid(Xnew, Xnew_cont, Xnew_disc, X_temp_cont, X_temp_disc,
                          clust, beta, sigma2, g_fun, sigma, inv_cov_covs, lambda, power, kappa, u, 
                          B0, m0, a0, b0, grid).t();
          prediction3.row(res_index) = pred_values_grid2(Xnew, Xnew_cont, Xnew_disc, X_temp_cont, X_temp_disc,
                          clust, beta, sigma2, g_fun, sigma, inv_cov_covs, lambda, power, kappa, u, 
                          B0, m0, a0, b0, grid).t();
        }
      }
      
      res_index +=1 ;
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
  results["m0"] = m0;
  results["B0"] = B0;
  results["a0"] = a0;
  results["b0"] = b0;
  results["kappa"] = kappa;
  results["sigma"] = sigma;
  results["u"] = u_result;
  results["s2v"] = s2v_result;
  results["beta"] = res_beta;
  results["sigma2"] = res_sigma2;
  results["clust"] = res_clust;
  results["prediction"] = prediction;
  results["prediction2"] = prediction2;
  results["prediction3"] = prediction3;
  
  return results;
}

