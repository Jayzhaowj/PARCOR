#ifndef __vec_to_mat__
#define __vec_to_mat__

#include <RcppArmadillo.h>
#include <Rcpp.h>

// [[Rcpp::depends(RcppArmadillo)]]

arma::mat vec_to_mat(arma::vec A, int nrow, int ncol){
  arma::mat B;
  B.insert_cols(0, A);
  B.reshape(nrow, ncol);
  return arma::trans(B);
}

#endif // __vec_to_mat__
