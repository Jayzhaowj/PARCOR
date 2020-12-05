#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "shared/filter_TVVAR.hpp"


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List filter_smooth_TVVAR(arma::mat F1, 
                         arma::mat G, 
                         arma::mat mk_0,
                         arma::mat Ck_0,
                         double n_0, 
                         arma::mat S_0,
                         int m,
                         arma::mat delta,
                         int pp
){
  int n_t = F1.n_cols;
  int n_I2 = G.n_rows;
  int delta_n = delta.n_rows;
  arma::mat mnt(n_I2, n_t, arma::fill::zeros);
  arma::cube Cnt(n_I2, n_I2, n_t);
    
    // estimate the innovation variance
  Rcpp::List temp_filter_opt = filter_TVVAR(F1, G, mk_0, Ck_0, n_0, S_0, m, delta.row(0), pp);
  
  double ll_max = temp_filter_opt["ll"];
  arma::rowvec delta_min = delta.row(0);
  for(int j = 1; j < delta_n; j++){
    Rcpp::List temp_filter_new = filter_TVVAR(F1, G, mk_0, Ck_0, n_0, S_0, m, delta.row(j), pp);
    double ll_new = temp_filter_new["ll"];
    if(ll_max < ll_new){
      temp_filter_opt = temp_filter_new;
      delta_min = delta.row(j); 
    }
    Rprintf("completation: %i / %i \r", j+1, delta_n);
  }
  arma::cube St = temp_filter_opt["St"];
  arma::mat at = temp_filter_opt["at"];
  arma::mat mt = temp_filter_opt["mt"];
  arma::cube F1t = temp_filter_opt["F1t"];
  arma::cube Ct = temp_filter_opt["Ct"];
  arma::cube Rt = temp_filter_opt["Rt"];
  mnt.col(n_t  - 1) = mt.col(n_t - 1);
  Cnt.slice(n_t - 1) = Ct.slice(n_t - 1);
  for(int i = (n_t - 2); i > (pp - 1); i--){
    Rt.slice(i+1) = 0.5*Rt.slice(i+1) + 0.5*arma::trans(Rt.slice(i+1));
    arma::mat Rtp1_inv = arma::inv_sympd(Rt.slice(i+1));
    arma::mat Bt = Ct.slice(i) * arma::trans(G) * Rtp1_inv;
    mnt.col(i) = mt.col(i) + Bt * (mnt.col(i+1) - at.col(i+1));
    Cnt.slice(i) = Ct.slice(i) - Bt * (Rt.slice(i+1) - Cnt.slice(i+1))*arma::trans(Bt);
  }
  return Rcpp::List::create(Rcpp::Named("mnt") = mnt, Rcpp::Named("St") = St, Rcpp::Named("filter_opt") = temp_filter_opt,
                            Rcpp::Named("ll_max") = ll_max, Rcpp::Named("delta_min") = delta_min);
}
