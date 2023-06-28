#include <RcppArmadillo.h>
#include <RcppDist.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

//----------------- UTILITY (PARTITIONS) ----------------------------------------------

//[[Rcpp::export]]
mat psm(mat M){
  // initialize results
  mat result(M.n_cols, M.n_cols, fill::zeros);
  
  for(uword i = 0; i < M.n_cols; i++){
    for(uword j = 0; j <= i; j++){
      result(i,j) = accu(M.col(i) == M.col(j));
      result(j,i) = result(i,j);
    }
    Rcpp::checkUserInterrupt();
  }
  return(result / M.n_rows);
}

//[[Rcpp::export]]
mat clean_partition(mat M){
  
  uvec index(M.n_cols);
  vec tvec(M.n_cols);
  // initialize results
  mat result(M.n_rows, M.n_cols, fill::zeros);
  
  // for each row
  for(uword k = 0; k < M.n_rows; k++){
    tvec = M.row(k).t();
    
    for(uword j = 0; j < max(M.row(k)); j++){
      while((accu(tvec == j + 1) == 0) && (accu(tvec > j + 1) != 0)){
        index = find(tvec > j + 1);
        tvec(index) = tvec(index) - 1;
      }
    }
    
    result.row(k) = tvec.t();
    Rcpp::checkUserInterrupt();
  }
  return(result);
}

//[[Rcpp::export]]
vec VI_LB(mat C_mat, mat psm_mat){
  
  vec result(C_mat.n_rows);
  double f = 0.0;
  int n = psm_mat.n_cols;
  vec tvec(n);
  
  for(uword j = 0; j < C_mat.n_rows; j++){
    f = 0.0;
    for(uword i = 0; i < n; i++){
      tvec = psm_mat.col(i);
      f += (log2(accu(C_mat.row(j) == C_mat(j,i))) +
        log2(accu(tvec)) -
        2 * log2(accu(tvec.elem(find(C_mat.row(j).t() == C_mat(j,i))))))/n;
    }
    result(j) = f;
    Rcpp::checkUserInterrupt();
  }
  return(result);
}