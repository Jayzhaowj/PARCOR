// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <math.h>
#include <RcppDist.h>
#include "shared/pDIC_hier.hpp"
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]
Rcpp::List forward_filter_backward_smooth(arma::mat yt, arma::mat F1, arma::mat F2,
                                          int n_t, int I, int m, int type, int P,
                                          double delta1, double delta2, int sample_size){
  // some constants
  int sign = 1;
  arma::mat I_n(I, I, arma::fill::eye);
  // initial states
  arma::colvec mk_0(I, arma::fill::zeros);
  arma::mat Ck_0(I, I, arma::fill::eye);
  //arma::mat Ck_s_0(I, I, arma::fill::eye);
  double n_0 = 1;
  double S_0 = 1;
  double ll = 0.0;
  arma::vec nt(n_t, arma::fill::zeros);
  arma::vec dt(n_t, arma::fill::zeros);
  arma::vec St(n_t, arma::fill::zeros);

  // parcor states
  arma::mat at(I, n_t, arma::fill::zeros);
  arma::mat mt(I, n_t, arma::fill::zeros);
  arma::mat ft(I, n_t, arma::fill::zeros);
  arma::mat et(I, n_t, arma::fill::zeros);

  arma::cube Rt(I, I, n_t, arma::fill::zeros);
  arma::cube Ct(I, I, n_t, arma::fill::zeros);
  arma::cube Ut(I, I, n_t, arma::fill::zeros);
  arma::cube Qt(I, I, n_t, arma::fill::zeros);
  arma::cube inv_Qt(I, I, n_t, arma::fill::zeros);

  //structure level
  arma::mat akt(I, n_t, arma::fill::zeros);
  arma::mat mkt(I, n_t, arma::fill::zeros);
  arma::cube Rkt(I, I, n_t, arma::fill::zeros);
  arma::cube Ckt(I, I, n_t, arma::fill::zeros);
  arma::cube V2t(I, I, n_t, arma::fill::zeros);
  arma::cube Ukt(I, I, n_t, arma::fill::zeros);
  arma::cube F1t(I, I, n_t, arma::fill::zeros);

  // smooth part
  arma::mat mnt(I, n_t, arma::fill::zeros);
  arma::cube Cnt(I, I, n_t, arma::fill::zeros);
  arma::mat mnkt(I, n_t, arma::fill::zeros);
  arma::cube Cnkt(I, I, n_t, arma::fill::zeros);
  arma::cube Ant(I, I, n_t, arma::fill::zeros);
  arma::cube Ankt(I, I, n_t, arma::fill::zeros);
  arma::cube Bnt(I, I, n_t, arma::fill::zeros);
  arma::cube Bnkt(I, I, n_t, arma::fill::zeros);
  arma::mat resid(I, n_t, arma::fill::zeros);


  // asign the lower bound and upper bound
  int ubound = 0;
  int lbound = 0;
  if(type == 1){
    ubound = n_t;
    lbound = m;
    sign = 1;
  }else{
    ubound = n_t - m;
    lbound = 0;
    sign = -1;
  }


  // filtering conditional on V:
  for(int i = lbound; i < ubound; i++){

    // prior distribution update
    F1t.slice(i) = arma::diagmat(F1.col(i - sign*m));
    if(i == lbound){
      akt.col(i) = mk_0;
      Rkt.slice(i) = 10*Ck_0/delta1;
      Rkt.slice(i) = 0.5*Rkt.slice(i) + 0.5*arma::trans(Rkt.slice(i));
    }else{
      akt.col(i) = mkt.col(i-1);
      Rkt.slice(i) = Ckt.slice(i-1)/delta1;
      Rkt.slice(i) = 0.5*Rkt.slice(i) + 0.5*arma::trans(Rkt.slice(i));
    }

    at.col(i) = F2*akt.col(i);
    Rt.slice(i) = F2*Rkt.slice(i)*arma::trans(F2)/delta2;
    Rt.slice(i) = 0.5*Rt.slice(i) + 0.5*arma::trans(Rt.slice(i));
    V2t.slice(i) = (1 - delta2)/delta2 * F2 * Rkt.slice(i) * arma::trans(F2);

    // predictive distribution update
    ft.col(i) = F1t.slice(i) * at.col(i);
    et.col(i) = yt.col(i) - ft.col(i);
    if(i == lbound){
      Qt.slice(i) = F1t.slice(i) * Rt.slice(i) * arma::trans(F1t.slice(i)) + I_n;
      Qt.slice(i) = 0.5 * Qt.slice(i) + 0.5 * arma::trans(Qt.slice(i));
    }else{
      Qt.slice(i) = F1t.slice(i) * Rt.slice(i) * arma::trans(F1t.slice(i)) + St(i-1)*I_n;
      Qt.slice(i) = 0.5 * Qt.slice(i) + 0.5 * arma::trans(Qt.slice(i));
    }

    // posterior distribution
    // evolution equation
    Ukt.slice(i) = Rkt.slice(i) * arma::trans(F1t.slice(i)*F2);
    inv_Qt.slice(i) = arma::inv_sympd(Qt.slice(i));

    if(i == lbound){
      nt(i) = n_0;
      St(i) = S_0;
      dt(i) = n_0*S_0;
    }else{
      nt(i) = nt(i-1) + I;
      dt(i) = dt(i-1) + St(i-1)*arma::as_scalar(arma::trans(et.col(i)) * inv_Qt.slice(i) * et.col(i));
      St(i) = dt(i)/nt(i);
    }
    mkt.col(i) = akt.col(i) + Ukt.slice(i) * inv_Qt.slice(i)*et.col(i);
    if(i == lbound){
      Ckt.slice(i) = St(i)*(Rkt.slice(i) - Ukt.slice(i)*inv_Qt.slice(i)*arma::trans(Ukt.slice(i)));
    }else{
      Ckt.slice(i) = (St(i)/St(i-1))*(Rkt.slice(i) - Ukt.slice(i)*inv_Qt.slice(i)*arma::trans(Ukt.slice(i)));
    }
    Ckt.slice(i) = 0.5*Ckt.slice(i) + 0.5*arma::trans(Ckt.slice(i));

    // Structural equation
    Ut.slice(i) = Rt.slice(i) * arma::trans(F1t.slice(i));
    mt.col(i) = at.col(i) + Ut.slice(i) * inv_Qt.slice(i) * et.col(i);
    if(i == lbound){
      Ct.slice(i) = St(i)*(Rt.slice(i) - Ut.slice(i) * inv_Qt.slice(i) * arma::trans(Ut.slice(i)));
    }else{
      Ct.slice(i) = (St(i)/St(i-1))*(Rt.slice(i) - Ut.slice(i) * inv_Qt.slice(i) * arma::trans(Ut.slice(i)));
    }
    Ct.slice(i) = 0.5*Ct.slice(i) + 0.5*arma::trans(Ct.slice(i));

    if((i >= P) & (i < n_t - P)){
      arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(i)), ft.col(i), Qt.slice(i), true);
      ll += arma::sum(tmp_ll);
    }

  }


  // smooth part
  mnt.col(ubound-1) = mt.col(ubound-1);
  Cnt.slice(ubound-1) = Ct.slice(ubound-1);
  mnkt.col(ubound-1) = mkt.col(ubound-1);
  Cnkt.slice(ubound-1) = Ckt.slice(ubound-1);
  for(int i = (ubound - 2); i > lbound - 1; i--){
    arma::mat V02t_s = St(ubound-1) * I_n + F1t.slice(i)*V2t.slice(i)*arma::trans(F1t.slice(i));
    arma::mat inv_V02t_s = arma::inv_sympd(0.5*V02t_s + 0.5*arma::trans(V02t_s));

    Ant.slice(i) = F2*Ckt.slice(i)*arma::trans((I_n - V2t.slice(i)*arma::trans(F1t.slice(i))*inv_V02t_s*F1t.slice(i))*F2);
    Ankt.slice(i) = Ckt.slice(i);

    arma::mat inv_Rtp1 = arma::inv_sympd(Rt.slice(i+1));
    arma::mat inv_Rktp1 = arma::inv_sympd(Rkt.slice(i+1));

    Bnkt.slice(i) = arma::trans(Ankt.slice(i))*inv_Rktp1;
    Bnt.slice(i) = arma::trans(Ant.slice(i))*inv_Rtp1;

    mnt.col(i) = mt.col(i) + Bnt.slice(i)*(mnt.col(i+1) - at.col(i+1));
    Cnt.slice(i) = Ct.slice(i) - Bnt.slice(i)*(Rt.slice(i+1) - Cnt.slice(i+1))*arma::trans(Bnt.slice(i));
    Cnt.slice(i) = 0.5*Cnt.slice(i) + 0.5*arma::trans(Cnt.slice(i));

    mnkt.col(i) = mkt.col(i) + Bnkt.slice(i)*(mnkt.col(i+1) - akt.col(i+1));
    Cnkt.slice(i) = Ckt.slice(i) - Bnkt.slice(i)*(Rkt.slice(i+1) - Cnkt.slice(i+1))*arma::trans(Bnkt.slice(i));
    Cnkt.slice(i) = 0.5*Cnkt.slice(i) + 0.5*arma::trans(Cnkt.slice(i));

  }

  // compute the residuals
  resid.col(ubound-1) = yt.col(ubound-1) - F1t.slice(ubound-1) * mnt.col(ubound-1);
  for(int i = (ubound - 2); i > lbound - 1; i--){
    resid.col(i) = yt.col(i) - F1t.slice(i) * mnt.col(i);
  }



  return Rcpp::List::create(Rcpp::Named("mnt") = mnt,
                            Rcpp::Named("mnkt") = mnkt,
                            Rcpp::Named("Cnt") = Cnt,
                            Rcpp::Named("Cnkt") = Cnkt,
                            //Rcpp::Named("mnt_sample") = mnt_sample,
                            //Rcpp::Named("mnkt_sample") = mnkt_sample,
                            Rcpp::Named("akt") = akt,
                            Rcpp::Named("Rkt") = Rkt,
                            Rcpp::Named("Rt") = Rt,
                            Rcpp::Named("residuals") = resid,
                            Rcpp::Named("nt") = nt,
                            Rcpp::Named("sigma2t") = St,
                            Rcpp::Named("Qt") = Qt,
                            Rcpp::Named("nt") = nt,
                            Rcpp::Named("ll") = ll);
}

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
Rcpp::List sample_parcor_hier(Rcpp::List result, int m, int P, int type,
                              int sample_size){
  arma::mat mnt = result["mnt"];
  arma::mat mnkt = result["mnkt"];
  arma::cube Cnt = result["Cnt"];
  arma::cube Cnkt = result["Cnkt"];
  arma::vec St = result["sigma2t"];
  arma::vec nt = result["nt"];
  int n_t = mnt.n_cols;
  int n_I = mnt.n_rows;

  arma::cube mnt_sample(sample_size, n_I, n_t, arma::fill::zeros);
  arma::cube mnkt_sample(sample_size, n_I, n_t, arma::fill::zeros);
  // asign the lower bound and upper bound
  int ubound = 0;
  int lbound = 0;
  if(type == 1){
    ubound = n_t;
    lbound = m;
  }else{
    ubound = n_t - m;
    lbound = 0;
  }



  for(int i = lbound; i < ubound; i++){
    mnt_sample.slice(i) = rmvt(sample_size, mnt.col(i), Cnt.slice(i), nt(i));
    mnkt_sample.slice(i) = rmvt(sample_size, mnkt.col(i), Cnkt.slice(i), nt(i));
  }
  return Rcpp::List::create(Rcpp::Named("mnt_sample") = mnt_sample,
                            Rcpp::Named("mnkt_sample") = mnkt_sample);
}



// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]
Rcpp::List ffbs_DIC(arma::mat yt, arma::mat F1, arma::mat F2,
                    int n_t, int I, int m, int type, int P,
                    arma::mat delta, bool DIC, int sample_size,
                    int chains, bool uncertainty){
  // do ffbs
  int delta_n = delta.n_rows;
  double ll_DIC = 0.0;
  double pDIC = 0.0;
  Rcpp::List result_opt = forward_filter_backward_smooth(yt, F1, F2,
                                                         n_t, I, m,  type, P,
                                                         delta(0, 0), delta(0, 1),
                                                         sample_size);
  double ll_max = result_opt["ll"];
  arma::rowvec delta_min = delta.row(0);

  for(int i = 1; i < delta_n; i++){
    Rcpp::List result_new = forward_filter_backward_smooth(yt, F1, F2,
                                                           n_t, I, m,  type, P,
                                                           delta(i, 0), delta(i, 1),
                                                           sample_size);
    double ll_new = result_new["ll"];
    if(ll_max < ll_new){
      result_opt = result_new;
      ll_max = ll_new;
      delta_min = delta.row(i);
    }
  }
  ll_DIC = ll_max;

  if(DIC){
    pDIC = compute_pDIC(result_opt, yt, F1, F2, sample_size,
                               m, P, type, chains, ll_DIC);
  }

  if(uncertainty){
    Rcpp::List sample = sample_parcor_hier(result_opt, m, P, type, sample_size);
    return Rcpp::List::create(Rcpp::Named("mnt") = result_opt["mnt"],
                              Rcpp::Named("mnkt") = result_opt["mnkt"],
                              Rcpp::Named("mnt_sample") = sample["mnt_sample"],
                              Rcpp::Named("mnkt_sample") = sample["mnkt_sample"],
                              Rcpp::Named("residuals") = result_opt["residuals"],
                              Rcpp::Named("sigma2t") = result_opt["sigma2t"],
                              Rcpp::Named("delta") = delta_min,
                              Rcpp::Named("ll") = ll_DIC,
                              Rcpp::Named("pDIC") = pDIC);
  }else{
    return Rcpp::List::create(Rcpp::Named("mnt") = result_opt["mnt"],
                              Rcpp::Named("mnkt") = result_opt["mnkt"],
                              Rcpp::Named("Cnt") = result_opt["Cnt"],
                              Rcpp::Named("Cnkt") = result_opt["Cnkt"],
                              Rcpp::Named("residuals") = result_opt["residuals"],
                              Rcpp::Named("sigma2t") = result_opt["sigma2t"],
                              Rcpp::Named("delta") = delta_min,
                              Rcpp::Named("ll") = ll_DIC,
                              Rcpp::Named("pDIC") = pDIC);
  }
}
