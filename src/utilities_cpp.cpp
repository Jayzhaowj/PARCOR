#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;

//[[Rcpp::depends(RcppArmadillo)]]

double compute_norm_matrix(const arma::mat& A_new, const arma::mat& A_old){
  if((arma::norm(arma::vectorise(A_old), 2) == 0) & (arma::norm(arma::vectorise(A_new), 2) == 0)){
    return 0;
  }else if(arma::norm(arma::vectorise(A_old), 2) == 0){
    return std::abs(arma::norm(arma::vectorise(A_new), 2) - arma::norm(arma::vectorise(A_old), 2))/arma::norm(arma::vectorise(A_new), 2);
  }else{
    return std::abs(arma::norm(arma::vectorise(A_new), 2) - arma::norm(arma::vectorise(A_old), 2))/arma::norm(arma::vectorise(A_old), 2);
  }

}

double compute_norm_vector(const arma::vec& A_new, const arma::vec& A_old){
  if((arma::norm(A_old, 2) == 0) & (arma::norm(A_new, 2) == 0)){
    return 0;
  }else if(arma::norm(A_old, 2) == 0){
    return std::abs(arma::norm(A_new, 2) - arma::norm(A_old, 2))/arma::norm(A_new, 2);
  }else{
    return std::abs(arma::norm(A_new, 2) - arma::norm(A_old, 2))/arma::norm(A_old, 2);
  }

}
