// PARCOR
// run_parcor is main function of performing PARCOR model
// misc about transformation from PARCOR to AR.
// gen_AR_sample is function of generating sample of AR coefficients.
// PARCOR_to_AR_fun is function of transforming from PARCOR to AR.

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <RcppDist.h>
#include "shared/gen_F1t.hpp"
#include "shared/pDIC_multivariate.hpp"
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]

Rcpp::List filter(arma::mat F1_fwd,
                  arma::mat F1_bwd,
                  arma::mat S_0,
                  int m,
                  arma::rowvec delta,
                  int type_num,
                  int P,
                  int n_t,
                  int n_I,
                  int n_I2
){
  double ll = 0.0;
  int sign = 1;
  arma::colvec mk_0(n_I2, arma::fill::zeros);
  arma::mat Ck_0(n_I2, n_I2, arma::fill::eye);
  double n_0 = 1.0;
  arma::dmat mt(n_I2, n_t, arma::fill::zeros);
  arma::mat F1t;
  Rcpp::List Ct(n_t);
  arma::mat Qt(n_I, n_I);
  arma::mat St_sqp(n_I, n_I, arma::fill::zeros);
  arma::mat St(n_I, n_I);
  //arma::dcube St(n_I, n_I, n_t);
  arma::dmat S_comp(n_I, n_I, arma::fill::zeros);
  arma::dmat At(n_I2, n_I, arma::fill::zeros);
  arma::dmat F1(n_I, n_t, arma::fill::zeros);
  arma::dmat yt(n_I, n_t, arma::fill::zeros);
  arma::colvec et;
  int ubound = 0;
  int lbound = 0;
  if(type_num == 1){
    ubound = n_t;
    lbound = P;
    mt.col(P - 1) = mk_0;
    Ct(P - 1) = Ck_0;
    St = S_0;
    F1 = F1_bwd;
    yt = F1_fwd;
  }else{
    ubound = n_t - P;
    lbound = 0;
    F1 = F1_fwd;
    yt = F1_bwd;
    sign = -1;
  }
  arma::dmat F1_new(n_I, n_t, arma::fill::zeros);
  arma::dmat delta_m = arma::diagmat(arma::pow(delta, -0.5));
  for(int i = lbound; i < ubound; i++){
    F1t = arma::trans(gen_Ft(F1.col(i - sign * m)));
    if(i == 0){
      Qt = F1t * (delta_m * Ck_0 * delta_m) * arma::trans(F1t) + S_0;
      St_sqp = arma::sqrtmat_sympd(S_0);
    }else{
      //Qt = F1t * (delta_m * Rcpp::as<arma::mat>(Ct(i-1)) * delta_m) * arma::trans(F1t) + St.slice(i-1);
      Qt = F1t * (delta_m * Rcpp::as<arma::mat>(Ct(i-1)) * delta_m) * arma::trans(F1t) + St;
      //St_sqp = arma::sqrtmat_sympd(St.slice(i-1));
      St_sqp = arma::sqrtmat_sympd(St);
    }


    if(!Qt.is_symmetric()){
      Qt = 0.5*Qt + 0.5*arma::trans(Qt);
      //arma::cout << "Qt is not symmetric \n" << arma::endl;
    }

    arma::dmat Qt_inv = arma::inv_sympd(Qt);

    if(!Qt_inv.is_symmetric()){
      Qt_inv = 0.5*Qt_inv + 0.5*arma::trans(Qt_inv);
    }

    arma::dmat Qt_inv_sq = arma::sqrtmat_sympd(Qt_inv);
    if(i == 0){
      At = (delta_m * Ck_0 * delta_m) * arma::trans(F1t) * Qt_inv;
      et = yt.col(i) - F1t * mk_0;
      mt.col(i) = mk_0 + At * et;
    }else{
      At = (delta_m * Rcpp::as<arma::mat>(Ct(i-1)) * delta_m) * arma::trans(F1t) * Qt_inv;
      et = yt.col(i) - F1t * mt.col(i-1);
      mt.col(i) = mt.col(i-1) + At * et;
    }

    S_comp += St_sqp * Qt_inv_sq * et * arma::trans(et) * Qt_inv_sq * St_sqp;
    //St.slice(i) = (n_0*S_0 + S_comp)/(n_0 + i + 1);
    St = (n_0*S_0 + S_comp)/(n_0 + i + 1);
    //St.slice(i) = 0.5*St.slice(i) + 0.5*arma::trans(St.slice(i));
    if(!St.is_symmetric()){
      St = 0.5 * St + 0.5 * arma::trans(St);
      //arma::cout << "St is not symmetric \n" << arma::endl;
    }
    if(i == 0){
      Ct(i) = (delta_m * Ck_0 * delta_m) - At * Qt * arma::trans(At);
    }else{
      Ct(i) = (delta_m * Rcpp::as<arma::mat>(Ct(i-1)) * delta_m) - At * Qt * arma::trans(At);
    }

    if((i >= P) & (i < n_t - P) ){
      arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(i)), F1t * mt.col(i-1), Qt, true);
      ll += arma::sum(tmp_ll);
    }
  }
    return Rcpp::List::create(Rcpp::Named("ll") = ll,
                              Rcpp::Named("mt") = mt,
                              Rcpp::Named("Ct") = Ct,
                              Rcpp::Named("St") = St);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List filter_smooth(arma::mat F1_fwd,
                         arma::mat F1_bwd,
                         arma::mat S_0,
                         int m,
                         arma::mat delta,
                         int type_num,
                         int P,
                         bool DIC,
                         int sample_size,
                         int chains,
                         bool uncertainty
){
  // initializing
    //std::cout << "a1";
    int n_t = F1_fwd.n_cols;  // the number of time points
    int n_I = F1_fwd.n_rows;  // the number of time series
    int n_I2 = std::pow(n_I, 2);  // the dimension of each stage
    int delta_n = delta.n_rows; // the number of choice of discount factors
    int lbound = 0;
    int ubound = 0;

    arma::dmat mnt(n_I2, n_t, arma::fill::zeros);
    Rcpp::List Cnt(n_t);
    arma::dmat F1(n_I, n_t, arma::fill::zeros);
    arma::dmat yt(n_I, n_t, arma::fill::zeros);
    arma::dmat F1_new(n_I, n_t, arma::fill::zeros);
    arma::rowvec ll(delta_n);
    double ll_DIC = 0.0;
    double es_DIC = 0.0;
    int sign = 1;
    if(type_num == 1){
      F1 = F1_bwd;
      yt = F1_fwd;
    }else{
      F1 = F1_fwd;
      yt = F1_bwd;
      sign = -1;
    }
    // do the filtering updating function
    Rcpp::List filter_opt = filter(F1_fwd, F1_bwd, S_0, m, delta.row(0), type_num, P, n_t, n_I, n_I2);
    double ll_max = filter_opt["ll"];
    arma::rowvec delta_min = delta.row(0);
    ll(0) = ll_max;

    for(int j = 1; j < delta_n; j++){
        Rcpp::List filter_new = filter(F1_fwd, F1_bwd, S_0, m, delta.row(j), type_num, P, n_t, n_I, n_I2);
        double ll_new = filter_new["ll"];
        ll(j) = ll_new;
        if(ll_max < ll_new){
            filter_opt = filter_new;
            ll_max = ll_new;
            delta_min = delta.row(j);
        }
      // Rprintf("completation: %i / %i \r", j+1, delta_n);
    }

    arma::mat mt = filter_opt["mt"];
    Rcpp::List Ct = filter_opt["Ct"];
    arma::mat Rt(n_I2, n_I2, arma::fill::zeros);
    //arma::cube St = filter_opt["St"];
    arma::mat F1t;

    if(type_num == 1){
        lbound = P;
        ubound = n_t;
    }else{
        lbound = 0;
        ubound = n_t - P;

    }
    // This following part is smoothing.
    // initializing.
    mnt.col(ubound - 1) = mt.col(ubound - 1);
    if(uncertainty){
      Cnt(ubound - 1) = Rcpp::as<arma::mat>(Ct(ubound - 1));
    }

    arma::mat delta_m = arma::diagmat(arma::pow(delta_min, -0.5));
    for(int i = (ubound - 2); i > (lbound - 1); i--){
        Rt = delta_m * Rcpp::as<arma::mat>(Ct(i)) * delta_m;

        if(!Rt.is_symmetric()){
          Rt = 0.5 * Rt + 0.5 * arma::trans(Rt);
          //arma::cout << "Rt is not symmetric \n" << arma::endl;
        }
        arma::mat Rtp1_inv = arma::inv_sympd(Rt);
        arma::mat Bt = Rcpp::as<arma::mat>(Ct(i)) * Rtp1_inv;
        mnt.col(i) = mt.col(i) + Bt * (mnt.col(i+1) - mt.col(i));
        if(uncertainty){
          Cnt(i) = Rcpp::as<arma::mat>(Ct(i)) - Bt * (Rt - Rcpp::as<arma::mat>(Cnt(i+1)))*arma::trans(Bt);
          Cnt(i) = 0.5*Rcpp::as<arma::mat>(Cnt(i)) + 0.5*arma::trans(Rcpp::as<arma::mat>(Cnt(i)));
        }
    }

    for(int i = lbound; i < ubound; i++){
        F1t = arma::trans(gen_Ft(F1.col(i - sign * m)));
        F1_new.col(i) = yt.col(i) - F1t * mnt.col(i); //update the next stage prediction error.
    }


    if(DIC){
        Rcpp::List tmp_DIC = compute_DIC(filter_opt, yt, F1, delta_min, sample_size, m, P, chains);
        ll_DIC = tmp_DIC["ll"];
        es_DIC = tmp_DIC["ES_mean"];
    }

    if(uncertainty){
          return Rcpp::List::create(Rcpp::Named("F1new") = F1_new,
                              Rcpp::Named("mnt") = mnt,
                              Rcpp::Named("Cnt") = Cnt,
                              Rcpp::Named("ll_max") = ll_max,
                              Rcpp::Named("delta_min") = delta_min,
                              //Rcpp::Named("filter_opt") = filter_opt,
                              Rcpp::Named("ll") = ll,
                              //Rcpp::Named("St") = St,
                              Rcpp::Named("St") =filter_opt["St"],
                              Rcpp::Named("ES_mean") = es_DIC,
                              Rcpp::Named("ll_DIC") = ll_DIC);
        }else{
                    return Rcpp::List::create(Rcpp::Named("F1new") = F1_new,
                              Rcpp::Named("mnt") = mnt,
                              Rcpp::Named("ll_max") = ll_max,
                              Rcpp::Named("delta_min") = delta_min,
                              Rcpp::Named("ll") = ll,
                              //Rcpp::Named("St") = St,
                              Rcpp::Named("St") =filter_opt["St"],
                              Rcpp::Named("ES_mean") = es_DIC,
                              Rcpp::Named("ll_DIC") = ll_DIC);
        }



}




