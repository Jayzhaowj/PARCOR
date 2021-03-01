// TV-VPARCOR model and hierarchical dynamic PARCOR model
//



#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <RcppDist.h>
#include "shared/whittle_algorithm.hpp"
#include "shared/dl_algorithm.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List run_whittle(arma::cube phi_fwd, arma::cube phi_bwd, int n_I){
  int n_t = phi_fwd.n_cols;
  int P = phi_fwd.n_slices;
  arma::cube akm_prev(n_I, n_t, P);
  arma::cube dkm_prev(n_I, n_t, P);
  Rcpp::List ar_coef_prev;
  Rcpp::List ar_coef(P);
  for(int i = 0; i < P; i++){
    if(i == 0){
      ar_coef(i) = whittle_algorithm(phi_fwd.slice(i), phi_bwd.slice(i),
                                     akm_prev, dkm_prev, n_I, i);
    }else{
      ar_coef_prev = ar_coef(i - 1);
      akm_prev = Rcpp::as<arma::cube>(ar_coef_prev["forward"]);
      dkm_prev = Rcpp::as<arma::cube>(ar_coef_prev["backward"]);
      ar_coef(i) = whittle_algorithm(phi_fwd.slice(i), phi_bwd.slice(i), akm_prev,
                                     dkm_prev, n_I, i);
    }
  }
  return ar_coef;
}



// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]

Rcpp::List  sample_tvar_coef(arma::cube phi_fwd,
                             arma::cube phi_bwd,
                             Rcpp::List Cnt_fwd,
                             Rcpp::List Cnt_bwd,
                             int n_I,
                             int P_opt,
                             int P_max,
                             int h
){
  int n_t = phi_fwd.n_cols;
  int n_I2 = n_I * n_I;
  arma::cube akm_prev(n_I, n_t, P_opt);
  arma::cube dkm_prev(n_I, n_t, P_opt);
  Rcpp::List ar_coef_prev;
  Rcpp::List ar_coef(P_opt);
  //initialize
  for(int i = 0; i < P_opt; i++){
    arma::mat phi_fwd_sample(n_I2, n_t, arma::fill::zeros);
    arma::mat phi_bwd_sample(n_I2, n_t, arma::fill::zeros);
    arma::mat mnt_fwd = phi_fwd.slice(i);
    arma::mat mnt_bwd = phi_bwd.slice(i);
    Rcpp::List Cnt_fwd_cur = Cnt_fwd(i);
    Rcpp::List Cnt_bwd_cur = Cnt_bwd(i);
    for(int j = n_t-1; j > (P_max-1); j--){
      arma::mat Cnt_fwd_cur_tmp = Rcpp::as<arma::mat>(Cnt_fwd_cur(j));
      phi_fwd_sample.col(j) = arma::trans(rmvnorm(1, mnt_fwd.col(j), Cnt_fwd_cur_tmp));
      if(j > (n_t-h-P_max-1)){
        phi_bwd_sample.col(j) = phi_fwd_sample.col(j);
      }else{
        arma::mat Cnt_bwd_cur_tmp = Rcpp::as<arma::mat>(Cnt_bwd_cur(j));
        phi_bwd_sample.col(j) = arma::trans(rmvnorm(1, mnt_bwd.col(j), Cnt_bwd_cur_tmp));
      }
    }
    if(i == 0){
      ar_coef(i) = whittle_algorithm(phi_fwd_sample, phi_bwd_sample, akm_prev, dkm_prev, n_I, i);
    }else{
      ar_coef_prev = ar_coef(i - 1);
      akm_prev = Rcpp::as<arma::cube>(ar_coef_prev["forward"]);
      dkm_prev = Rcpp::as<arma::cube>(ar_coef_prev["backward"]);
      ar_coef(i) = whittle_algorithm(phi_fwd_sample, phi_bwd_sample, akm_prev, dkm_prev, n_I, i);
    }
  }
  return ar_coef;
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List run_dl(arma::cube phi_fwd, arma::cube phi_bwd){
  int n_t = phi_fwd.n_cols;
  int P = phi_fwd.n_slices;
  int n_I = phi_fwd.n_rows;
  arma::cube akm_prev(n_I, n_t, P);
  arma::cube dkm_prev(n_I, n_t, P);
  Rcpp::List ar_coef_prev;
  Rcpp::List ar_coef(P);
  for(int i = 0; i < P; i++){
    if(i == 0){
      ar_coef(i) = dl_algorithm(phi_fwd.slice(i), phi_bwd.slice(i), akm_prev, dkm_prev, i);
    }else{
      ar_coef_prev = ar_coef(i - 1);
      akm_prev = Rcpp::as<arma::cube>(ar_coef_prev["forward"]);
      dkm_prev = Rcpp::as<arma::cube>(ar_coef_prev["backward"]);
      ar_coef(i) = dl_algorithm(phi_fwd.slice(i), phi_bwd.slice(i), akm_prev, dkm_prev, i);
    }
  }
  return ar_coef;
}
