#ifndef __filter_TVVAR__
#define __filter_TVVAR__

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <RcppDist.h>
#include "gen_Ft_TVVAR.hpp"

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]


Rcpp::List filter_TVVAR(arma::mat F1,
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
  int n_I = F1.n_rows;
  int n_I2 = G.n_rows;
  double ll = 0.0;
  arma::mat at(n_I2, n_t, arma::fill::zeros);
  arma::cube Rt(n_I2, n_I2, n_t);
  arma::mat mt(n_I2, n_t, arma::fill::zeros);
  arma::cube Ct(n_I2, n_I2, n_t);
  arma::cube F1t(n_I, n_I2, n_t);
  arma::cube Qt(n_I, n_I, n_t);
  arma::cube St(n_I, n_I, n_t);
  arma::mat S_comp(n_I, n_I, arma::fill::zeros);
  arma::mat ft(n_I, n_t, arma::fill::zeros);
  arma::mat Qt_eigvec;
  arma::vec Qt_eigval;
  arma::mat St_eigvec;
  arma::vec St_eigval;
  arma::mat At;
  arma::mat mnt(n_I2, n_t, arma::fill::zeros);
  arma::cube Cnt(n_I2, n_I2, n_t);
  mt.col(m - 1) = mk_0;
  Ct.slice(m - 1) = Ck_0;
  St.slice(m - 1) = S_0;
  for(int i = m; i < n_t; i++){
      F1t.slice(i) = arma::trans(gen_Ft_TVVAR(F1(arma::span::all, arma::span(i-m, i-1))));
      at.col(i) = G*mt.col(i-1);
      arma::mat delta_m = arma::diagmat(arma::pow(delta, -0.5));
      Rt.slice(i) = delta_m * G * Ct.slice(i-1) * arma::trans(G) * delta_m;
      Rt.slice(i) = 0.5*Rt.slice(i) + 0.5*arma::trans(Rt.slice(i));
      ft.col(i) = F1t.slice(i) *at.col(i);
      Qt.slice(i) = F1t.slice(i) * Rt.slice(i) * arma::trans(F1t.slice(i)) + St.slice(i-1);
      Qt.slice(i) = 0.5*Qt.slice(i) + 0.5*arma::trans(Qt.slice(i));
      arma::mat Qt_inv = arma::inv_sympd(Qt.slice(i));
      arma::mat Qt_inv_sq = arma::sqrtmat_sympd(Qt_inv);
      At = Rt.slice(i) * arma::trans(F1t.slice(i)) * Qt_inv;
      arma::colvec et = F1.col(i) - ft.col(i);
      arma::mat St_sqp = arma::sqrtmat_sympd(St.slice(i-1));
      S_comp += St_sqp * Qt_inv_sq * et * arma::trans(et) * Qt_inv_sq * St_sqp;
      St.slice(i) = (n_0*S_0 + S_comp)/(n_0 + i + 1);
      St.slice(i) = 0.5*St.slice(i) + 0.5*arma::trans(St.slice(i));
      Ct.slice(i) = Rt.slice(i) - At * Qt.slice(i) * arma::trans(At);
      mt.col(i) = at.col(i) + At * et;
      if(i >= pp){
          arma::vec tmp_ll = dmvnorm(arma::trans(F1.col(i)),ft.col(i), Qt.slice(i), true);
          ll += arma::sum(tmp_ll);
      }
  }
  return Rcpp::List::create(Rcpp::Named("ll") = ll, Rcpp::Named("mt") = mt, Rcpp::Named("Ct") = Ct,
                            Rcpp::Named("Rt") = Rt, Rcpp::Named("at") = at, Rcpp::Named("ft") = ft,
                            Rcpp::Named("F1t") = F1t, Rcpp::Named("yt") = F1, Rcpp::Named("Qt") = Qt,
                            Rcpp::Named("St") = St);
}


#endif // __filter_TVVAR__
