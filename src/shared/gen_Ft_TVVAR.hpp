#ifndef __gen_Ft_TVVAR__
#define __gen_Ft_TVVAR__

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat gen_Ft_TVVAR(arma::mat F1) {
    int n = F1.n_rows;
    arma::mat F1_flip = arma::fliplr(F1);
    arma::vec F1_vec = arma::vectorise(F1_flip);
    arma::mat temp_I;
    temp_I.eye(n, n);
    arma::mat F1t = arma::kron(F1_vec, temp_I);
    return F1t;
}

#endif // __gen_Ft_TVVAR__
