#include "RcppArmadillo.h"
#include "RcppDist.h"
using namespace arma;
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

/* 
  Evaluate the Mahalanobis distance while including a new
    observation in a cluster (old and new distance)
*/ 

void cont_dist_increment(mat Y,
                         vec X,
                         mat inv_cov_centroid,
                         vec &temp,
                         vec &temp2){
  
  rowvec coef_mean, coef_mean2;
  coef_mean = mean(Y, 0);
  coef_mean2 = (coef_mean * Y.n_rows + X.t()) / (Y.n_rows + 1); 
  temp.resize(Y.n_rows);
  temp2.resize(Y.n_rows + 1);
  
  for(uword r = 0; r < Y.n_rows; r++){
    temp(r) = as_scalar((Y.row(r) - coef_mean) * inv_cov_centroid * (Y.row(r) - coef_mean).t());
    
    temp2(r) = as_scalar((Y.row(r) - coef_mean2) * inv_cov_centroid * (Y.row(r) - coef_mean2).t());
  }
  
  temp2(temp2.n_elem - 1) = as_scalar((X.t() - coef_mean2) * inv_cov_centroid * (X.t() - coef_mean2).t());
}

/* 
  Evaluate the Hamming distance while including a new
    observation in a cluster (old and new distance)
*/ 

void hamming_dist_increment(mat Y,
                            vec X,
                            vec &temp,
                            vec &temp2){
  
  double k = (double) Y.n_cols; 
  temp.resize(Y.n_rows);
  temp2.resize(Y.n_rows + 1);
  temp.fill(0.0);
  temp2.fill(0.0);
  rowvec coef_mean, coef_mean2;
  coef_mean = round(mean(Y, 0));
  coef_mean2 = round((coef_mean * Y.n_rows + X.t()) / (Y.n_rows + 1)); 
  
  for(uword r = 0; r < Y.n_rows; r++){
    for(uword l = 0; l < Y.n_cols; l++){
      
      if(Y(r,l) != coef_mean(l)){
        temp(r) += 1.0 / k;
      }
      
      if(Y(r,l) != coef_mean2(l)){
        temp2(r) += 1.0 / k;
      }
    }
  }
  
  for(uword l = 0; l < k; l++){
    if(X(l) != coef_mean(l)){
      temp2(temp2.n_elem - 1) += 1.0 / k;
    }
  }
}

/* 
   compute the MC estimate of the avg distance increment
   the function returns a matrix where the first column is the distance increment
   and the second column is the sample size
*/

// [[Rcpp::export]]
mat MC_routine(mat Y,
               int nrep,
               uvec type_of_var, 
               mat inv_cov_covs){
  
  // initialize the quantities
  mat tY;
  mat inc_res(nrep,2);
  vec tval; 
  uvec tind;
  vec old_cont, old_disc, new_cont, new_disc;
  uword tnew; 
  double tdist_new, tdist_old;
  double wD = (double) accu(type_of_var == 0) / ((double) type_of_var.n_elem);
  double wC = 1 - wD;
  int mk = Y.n_rows - 1;
  int sample_size;
  
  // check which distance is needed
  bool use_cont = false;
  bool use_disc = false;
  if(accu(type_of_var == 0) != 0){
    use_disc = true;
  }
  if(accu(type_of_var == 1) != 0){
    use_cont = true;
  }
  
  // loop over nrep distinct samples
  for(uword rep = 0; rep < nrep; rep++){
    
    // sample a size, and a subset of elements
    sample_size = randi(distr_param(2, Y.n_rows));
    tind = conv_to<uvec>::from(randi(sample_size, distr_param(0, mk)));
    tnew = randi(distr_param(0, Y.n_rows - 1));
    
    // compute the distance and the increment
    new_cont.resize(sample_size + 1);
    new_disc.resize(sample_size + 1);
    old_cont.resize(sample_size);
    old_disc.resize(sample_size);
    
    new_cont.fill(0.0);
    new_disc.fill(0.0);
    old_cont.fill(0.0);
    old_disc.fill(0.0);
    
    tY = Y.rows(tind);
    tval = Y.row(tnew).t();
    
    if(use_cont){
      cont_dist_increment(tY.cols(find(type_of_var == 1)), tval.elem(find(type_of_var == 1)), inv_cov_covs, old_cont, new_cont);  
    }
    if(use_disc){
      hamming_dist_increment(tY.cols(find(type_of_var == 0)), tval.elem(find(type_of_var == 0)), old_disc, new_disc); 
    }
    
    tdist_new = accu(wC * new_cont + wD * new_disc);
    tdist_old = accu(wC * old_cont + wD * old_disc);
    
    // save the results 
    inc_res(rep,0) = tdist_new - tdist_old;
    inc_res(rep,1) = sample_size;
  }
  
  return(inc_res);
}

