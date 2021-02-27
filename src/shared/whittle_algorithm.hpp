#ifndef __whittle_algorithm__
#define __whittle_algorithm__

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "vec_to_mat.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

Rcpp::List whittle_algorithm(arma::mat phi_forward, arma::mat phi_backward,
                             arma::cube akm_prev, arma::cube dkm_prev,
                             int n_I, int cur_level){
  int n_t = phi_forward.n_cols;
  int n_I2 = phi_forward.n_rows;
  arma::cube akm_cur(n_I2, n_t, cur_level + 1);
  arma::cube dkm_cur(n_I2, n_t, cur_level + 1);
  if (cur_level == 0){
    akm_cur.slice(0) = phi_forward;
    dkm_cur.slice(0) = phi_backward;
  }
  else
  {
    akm_cur.slice(cur_level) = phi_forward;
    dkm_cur.slice(cur_level) = phi_backward;
    for(int i = 0; i < cur_level; i++){
      arma::mat akm_temp(n_I2, n_t);
      arma::mat dkm_temp(n_I2, n_t);
      arma::mat akm_i_prev = akm_prev.slice(i);
      arma::mat dkm_i_prev = dkm_prev.slice(i);
      arma::mat akmm_i_prev = akm_prev.slice(cur_level - i - 1);
      arma::mat dkmm_i_prev = dkm_prev.slice(cur_level - i - 1);
        for(int j = 0; j < n_t; j++){
        akm_temp.col(j) = akm_i_prev.col(j) - arma::trans(arma::vectorise(vec_to_mat(phi_forward.col(j), n_I, n_I) *  vec_to_mat(dkmm_i_prev.col(j), n_I, n_I), 1));
        dkm_temp.col(j) = dkm_i_prev.col(j) - arma::trans(arma::vectorise(vec_to_mat(phi_backward.col(j), n_I, n_I) * vec_to_mat(akmm_i_prev.col(j), n_I, n_I), 1));
      }
      akm_cur.slice(i) = akm_temp;
      dkm_cur.slice(i) = dkm_temp;
    }
  }
  return Rcpp::List::create(Rcpp::Named("forward") = akm_cur, Rcpp::Named("backward") = dkm_cur);
}


#endif // __whittle_algorithm__
