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
                                          double n_0, double S_0,
                                          int n_t, int n_I, int m, int type, int P,
                                          double delta1, double delta2, int sample_size){
  // some constants
  int sign = 1;
  arma::mat I_n(n_I, n_I, arma::fill::eye);
  // initial states
  arma::colvec mk_0(n_I, arma::fill::zeros);
  arma::mat Ck_0(n_I, n_I, arma::fill::eye);
  //arma::mat Ck_s_0(n_I, n_I, arma::fill::eye);
  // double n_0 = 1;
  // double S_0 = 1;
  double ll = 0.0;
  arma::vec nt(n_t, arma::fill::zeros);
  arma::vec dt(n_t, arma::fill::zeros);
  arma::vec St(n_t, arma::fill::zeros);

  // parcor states
  arma::mat at(n_I, n_t, arma::fill::zeros);
  arma::mat mt(n_I, n_t, arma::fill::zeros);
  arma::mat ft(n_I, n_t, arma::fill::zeros);
  arma::mat et(n_I, n_t, arma::fill::zeros);

  arma::cube Rt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Ct(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Ut(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Qt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube inv_Qt(n_I, n_I, n_t, arma::fill::zeros);

  //structure level
  arma::mat akt(n_I, n_t, arma::fill::zeros);
  arma::mat mkt(n_I, n_t, arma::fill::zeros);
  arma::cube Rkt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Ckt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube V2t(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Ukt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube F1t(n_I, n_I, n_t, arma::fill::zeros);

  // smooth part
  arma::mat mnt(n_I, n_t, arma::fill::zeros);
  arma::cube Cnt(n_I, n_I, n_t, arma::fill::zeros);
  arma::mat mnkt(n_I, n_t, arma::fill::zeros);
  arma::cube Cnkt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Ant(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Ankt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Bnt(n_I, n_I, n_t, arma::fill::zeros);
  arma::cube Bnkt(n_I, n_I, n_t, arma::fill::zeros);
  arma::mat resid(n_I, n_t, arma::fill::zeros);


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
      Rkt.slice(i) = S_0*Ck_0/delta1;
    }else{
      akt.col(i) = mkt.col(i-1);
      Rkt.slice(i) = Ckt.slice(i-1)/delta1;
    }
    Rkt.slice(i) = 0.5*Rkt.slice(i) + 0.5*arma::trans(Rkt.slice(i));
    at.col(i) = F2*akt.col(i);
    Rt.slice(i) = F2*Rkt.slice(i)*arma::trans(F2)/delta2;
    Rt.slice(i) = 0.5*Rt.slice(i) + 0.5*arma::trans(Rt.slice(i));
    if(i == lbound){
      V2t.slice(i) = ((1 - delta2)/delta2) * F2 * Rkt.slice(i) * arma::trans(F2)/S_0;
    }else{
      V2t.slice(i) = ((1 - delta2)/delta2) * F2 * Rkt.slice(i) * arma::trans(F2)/St(i-1);
    }


    // predictive distribution update
    ft.col(i) = F1t.slice(i) * at.col(i);
    et.col(i) = yt.col(i) - ft.col(i);
    if(i == lbound){
      Qt.slice(i) = F1t.slice(i) * Rt.slice(i) * arma::trans(F1t.slice(i)) + S_0*I_n;
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
      nt(i) = n_0 + n_I;
      dt(i) = n_0*S_0 + S_0*arma::as_scalar(arma::trans(et.col(i)) * inv_Qt.slice(i) * et.col(i));
    }else{
      nt(i) = nt(i-1) + n_I;
      dt(i) = dt(i-1) + St(i-1)*arma::as_scalar(arma::trans(et.col(i)) * inv_Qt.slice(i) * et.col(i));
    }
    St(i) = dt(i)/nt(i);
    mkt.col(i) = akt.col(i) + Ukt.slice(i) * inv_Qt.slice(i)*et.col(i);
    if(i == lbound){
      Ckt.slice(i) = (St(i)/S_0)*(Rkt.slice(i) - Ukt.slice(i)*inv_Qt.slice(i)*arma::trans(Ukt.slice(i)));
    }else{
      Ckt.slice(i) = (St(i)/St(i-1))*(Rkt.slice(i) - Ukt.slice(i)*inv_Qt.slice(i)*arma::trans(Ukt.slice(i)));
    }
    Ckt.slice(i) = 0.5*Ckt.slice(i) + 0.5*arma::trans(Ckt.slice(i));

    // Structural equation
    Ut.slice(i) = Rt.slice(i) * arma::trans(F1t.slice(i));
    mt.col(i) = at.col(i) + Ut.slice(i) * inv_Qt.slice(i) * et.col(i);
    if(i == lbound){
      Ct.slice(i) = (St(i)/S_0)*(Rt.slice(i) - Ut.slice(i) * inv_Qt.slice(i) * arma::trans(Ut.slice(i)));
    }else{
      Ct.slice(i) = (St(i)/St(i-1))*(Rt.slice(i) - Ut.slice(i) * inv_Qt.slice(i) * arma::trans(Ut.slice(i)));
    }
    Ct.slice(i) = 0.5*Ct.slice(i) + 0.5*arma::trans(Ct.slice(i));

    if((i >= P) & (i < n_t - P)){
      //arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(i)), ft.col(i), Qt.slice(i), true);
      //arma::vec tmp_ll = dmvt(arma::trans(yt.col(i)), ft.col(i), Qt.slice(i), nt(i-1), true);
      arma::vec tmp_ll = dmvnrm_arma_fast(arma::trans(yt.col(i)),
                                          arma::trans(ft.col(i)), Qt.slice(i), true);
      if(!tmp_ll.is_finite()){
        //Rcpp::Rcout << "Is Qt positive definite: " << (Qt.slice(i)).is_sympd() << "\n";
        Rcpp::Rcout << "ll: " << tmp_ll << "\n";
        //Rcpp::Rcout << "ft: " << ft.col(i) << "\n";
        //Rcpp::Rcout << "yt: " << yt.col(i) << "\n";
        Rcpp::Rcout << "iteration: " << i << "\n";
      }
      ll += arma::sum(tmp_ll);
    }

  }


  // smooth part
  mnt.col(ubound-1) = mt.col(ubound-1);
  Cnt.slice(ubound-1) = Ct.slice(ubound-1);
  mnkt.col(ubound-1) = mkt.col(ubound-1);
  Cnkt.slice(ubound-1) = Ckt.slice(ubound-1);
  for(int i = (ubound - 2); i > lbound - 1; i--){
    arma::mat V02t_s = I_n + F1t.slice(i)*V2t.slice(i)*arma::trans(F1t.slice(i));
    arma::mat inv_V02t_s = arma::inv_sympd(0.5*V02t_s + 0.5*arma::trans(V02t_s));

    Ant.slice(i) = F2*Ckt.slice(i)*arma::trans((I_n - V2t.slice(i)*arma::trans(F1t.slice(i))*inv_V02t_s*F1t.slice(i))*F2);
    Ankt.slice(i) = Ckt.slice(i);

    arma::mat inv_Rtp1 = arma::inv_sympd(Rt.slice(i+1));
    arma::mat inv_Rktp1 = arma::inv_sympd(Rkt.slice(i+1));

    Bnkt.slice(i) = arma::trans(Ankt.slice(i))*inv_Rktp1;
    Bnt.slice(i) = arma::trans(Ant.slice(i))*inv_Rtp1;

    mnt.col(i) = mt.col(i) + Bnt.slice(i)*(mnt.col(i+1) - at.col(i+1));
    Cnt.slice(i) = St(ubound-1)/St(i)*(Ct.slice(i) - Bnt.slice(i)*(Rt.slice(i+1) - Cnt.slice(i+1)*St(i)/St(ubound-1))*arma::trans(Bnt.slice(i)));
    Cnt.slice(i) = 0.5*Cnt.slice(i) + 0.5*arma::trans(Cnt.slice(i));

    mnkt.col(i) = mkt.col(i) + Bnkt.slice(i)*(mnkt.col(i+1) - akt.col(i+1));
    Cnkt.slice(i) = St(ubound-1)/St(i)*(Ckt.slice(i) - Bnkt.slice(i)*(Rkt.slice(i+1) - Cnkt.slice(i+1)*St(i)/St(ubound-1))*arma::trans(Bnkt.slice(i)));
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
                            Rcpp::Named("mt") = mt,
                            Rcpp::Named("Ct") = Ct,
                            Rcpp::Named("akt") = akt,
                            Rcpp::Named("Rkt") = Rkt,
                            Rcpp::Named("Rt") = Rt,
                            Rcpp::Named("residuals") = resid,
                            Rcpp::Named("nt") = nt,
                            Rcpp::Named("sigma2t") = St,
                          //Rcpp::Named("V2t") = V2t,
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
  arma::cube Ct = result["Ct"];
  arma::vec nt = result["nt"];

  int n_t = mnt.n_cols;
  int n_I = mnt.n_rows;
  arma::mat I_n(n_I, n_I, arma::fill::eye);
  arma::cube mnt_sample(sample_size, n_I, n_t, arma::fill::zeros);
  arma::cube mnkt_sample(sample_size, n_I, n_t, arma::fill::zeros);
  // asign the lower bound and upper bound
  int ubound = 0;
  int lbound = 0;
  if(type == 1){
    ubound = n_t;
    lbound = P;
  }else{
    ubound = n_t - P;
    lbound = 0;
  }



  for(int i = lbound; i < ubound; i++){
    try{
      mnt_sample.slice(i) = rmvt(sample_size, mnt.col(i), Cnt.slice(i), nt(ubound-1));
    }catch(...){
      //mnt_sample.slice(i) = rmvt(sample_size, mnt.col(i), Ct.slice(i), nt(ubound-1));
      Rprintf("\n The sampler failed at iteration %i", i);

    }

    mnkt_sample.slice(i) = rmvt(sample_size, mnkt.col(i), Cnkt.slice(i), nt(ubound-1));
  }
  return Rcpp::List::create(Rcpp::Named("mnt_sample") = mnt_sample,
                            Rcpp::Named("mnkt_sample") = mnkt_sample);
}



// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]
Rcpp::List ffbs_DIC(arma::mat yt, arma::mat F1, arma::mat F2,
                    double n_0, double S_0,
                    int n_t, int n_I, int m, int type, int P,
                    arma::mat delta, bool DIC, int sample_size,
                    int chains, bool uncertainty){
  // do ffbs
  int delta_n = delta.n_rows;
  double ll_DIC = 0.0;
  double pDIC = 0.0;
  Rcpp::List result_opt = forward_filter_backward_smooth(yt, F1, F2, n_0, S_0,
                                                         n_t, n_I, m,  type, P,
                                                         delta(0, 0), delta(0, 1),
                                                         sample_size);
  double ll_max = result_opt["ll"];
  arma::rowvec delta_min = delta.row(0);

  for(int i = 1; i < delta_n; i++){
    Rcpp::List result_new = forward_filter_backward_smooth(yt, F1, F2, n_0, S_0,
                                                           n_t, n_I, m,  type, P,
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
                              //Rcpp::Named("V2t") = result_opt["V2t"],
                              Rcpp::Named("delta") = delta_min,
                              Rcpp::Named("ll") = ll_DIC,
                              Rcpp::Named("pDIC") = pDIC);
  }else{
    return Rcpp::List::create(Rcpp::Named("mnt") = result_opt["mnt"],
                              Rcpp::Named("mnkt") = result_opt["mnkt"],
                              //Rcpp::Named("Cnt") = result_opt["Cnt"],
                              //Rcpp::Named("Cnkt") = result_opt["Cnkt"],
                              Rcpp::Named("residuals") = result_opt["residuals"],
                              Rcpp::Named("sigma2t") = result_opt["sigma2t"],
                              //Rcpp::Named("V2t") = result_opt["V2t"],
                              Rcpp::Named("delta") = delta_min,
                              Rcpp::Named("ll") = ll_DIC,
                              Rcpp::Named("pDIC") = pDIC);
  }
}


// [Rcpp::depends(RcppArmadillo)]
void update_beta_tilde(arma::mat& beta_nc,
                       arma::mat& beta2_nc,
                       arma::cube& beta_nc_cov,
                       arma::vec& y, arma::mat& x,
                       const arma::vec& theta_sr,
                       const arma::vec& beta_mean, const int N,
                       const double S_0,
                       arma::vec& St){
  // initialization
  int d = x.n_cols;

  arma::mat mt(d, N+1, arma::fill::zeros);
  //arma::cube Rt(d, d, N+1, arma::fill::zeros);
  arma::cube Ct(d, d, N+1, arma::fill::zeros);
  arma::mat mT(d, N+1, arma::fill::zeros);
  arma::cube CT(d, d, N+1, arma::fill::zeros);

  arma::vec St_tmp(N+1, arma::fill::zeros);

  //double St_sq;
  //double S_comp = 0.0;


  arma::mat theta_sr_diag = arma::diagmat(theta_sr);
  arma::vec yt_star = y - x*beta_mean;
  arma::mat Ft = x*theta_sr_diag;
  // if(Ft.has_nan()){
  //   Rcout << "Ft has nan!" << "\n";
  //   Rcout << "theta_sr_diag: " << theta_sr_diag.has_nan() << "\n";
  // }

  arma::mat I_d = arma::eye(d, d);
  arma::vec ft(N, arma::fill::zeros);
  double Qt;

  //double Qt_inv_sq;
  arma::vec At;
  double et;
  Ct.slice(0) = I_d;
  double n_0 = 1.0;
  St_tmp(0) = S_0;
  arma::mat Bt;
  arma::mat Rtp1_inv;

  for(int t = 1; t < N+1; t++){

    //Rt.slice(t) = Ct.slice(t-1) + I_d;
    //Rt.slice(t) = 0.5*Rt.slice(t) + 0.5*arma::trans(Rt.slice(t));
    ft(t-1) = arma::as_scalar(Ft.row(t-1)*mt.col(t-1));
    //Qt = arma::as_scalar(Ft.row(t-1)*Rt.slice(t)*arma::trans(Ft.row(t-1)) + St_tmp(t-1));
    Qt = arma::as_scalar(Ft.row(t-1)*(Ct.slice(t-1) + I_d)*arma::trans(Ft.row(t-1)) + St_tmp(t-1));
    //St_sq = std::sqrt(St_tmp(t-1));
    //Qt_inv_sq = std::sqrt(1.0/Qt);
    et = yt_star(t-1) - ft(t-1);
    //S_comp += St_sq * Qt_inv_sq * et * et * Qt_inv_sq * St_sq;
    n_0 += 1;
    St_tmp(t) = St_tmp(t-1) + St_tmp(t-1)/n_0*(et*et/Qt - 1);
    //S_comp += St_tmp(t-1) *  et * et / Qt;
    //St_tmp(t) = (n_0*S_0 + S_comp)/(n_0 + t);
    // if(std::isnan(St_tmp(t))){
    //   Rcout << "t: " << t << "\n";
    //   Rcout << "n_0*S_0: " << n_0*S_0 << "\n";
    //   Rcout << "S_comp: " << S_comp << "\n";
    //   Rcout << "St_tmp: " << St_tmp(t) << "\n";
    //   Rcout << "Qt_inv_sq: " << Qt_inv_sq << "\n";
    //   Rcout << "Qt: " << Qt << "\n";
    //   Rcout << "rest: " << arma::as_scalar(Ft.row(t-1)*Rt.slice(t)*arma::trans(Ft.row(t-1))) << "\n";
    //   Rcout << "Rt: " << (Rt.slice(t)).has_nan() << "\n";
    //   Rcout << "Rtm1: " << (Rt.slice(t-1)).has_nan() << "\n";
    //   Rcout << "Atm1: " << At.has_nan() << "\n";
    //   Rcout << "Ctm1: " << (Ct.slice(t-1)).has_nan() << "\n";
    //   break;
    // }
    //At = Rt.slice(t)*arma::trans(Ft.row(t-1))/Qt;
    At = (Ct.slice(t-1) + I_d)*arma::trans(Ft.row(t-1))/Qt;
    mt.col(t) = mt.col(t-1) + At*et;
    Ct.slice(t) = (St_tmp(t)/St_tmp(t-1))*(Ct.slice(t-1) + I_d - At*Qt*arma::trans(At));
    Ct.slice(t) = 0.5*Ct.slice(t) + 0.5*arma::trans(Ct.slice(t));
    // if((Ct.slice(t)).has_nan()){
    //   Rcout << "t: " << t << "\n";
    //   Rcout << "Rt: " << (Rt.slice(t)).has_nan() << "\n";
    //   Rcout << "Qt: " << Qt << "\n";
    //   Rcout << "res: " << arma::as_scalar(Ft.row(t-1)*Rt.slice(t)*arma::trans(Ft.row(t-1))) << "\n";
    //   Rcout << "St_tmp: " << St_tmp(t-1) << "\n";
    //   Rcout << "Ctm1: " << (Ct.slice(t-1)).has_nan() << "\n";
    //   Rcout << "Rt is" << std::endl << Rt.slice(t) << std::endl;
    //   Rcout << "Ft is" << std::endl << Ft.row(t-1) << std::endl;
    // }
  }
  St = St_tmp.rows(1, N);
  beta_nc.row(N) = arma::trans(mt.col(N));
  CT.slice(N) = Ct.slice(N);
  CT.slice(N) = 0.5*CT.slice(N) + 0.5*arma::trans(CT.slice(N));
  arma::mat tmp = CT.slice(N) + arma::trans(beta_nc.row(N))*beta_nc.row(N);
  beta2_nc.row(N) = arma::trans(tmp.diag());
  tmp.diag().zeros();
  beta_nc_cov.row(N) = tmp;
  //beta_nc_samp.col(N) = mT.col(N);
  y(N-1) = arma::as_scalar(yt_star(N-1) - Ft.row(N-1)*arma::trans(beta_nc.row(N)));
  for(int t = N-1; t >= 0; t--){
    //Rtp1_inv = arma::inv(Rt.slice(t+1));
    //if(!(Rt.slice(t+1)).is_sympd()){
    //  Rt.slice(t+1) = .5*Rt.slice(t+1) + 0.5*arma::trans(Rt.slice(t+1));
    //}
    //Rt.slice(t+1) = .5*Rt.slice(t+1) + 0.5*arma::trans(Rt.slice(t+1));
    //Rtp1_inv = arma::inv_sympd(Rt.slice(t+1));
    //Rtp1_inv = arma::inv(Rt.slice(t+1));
    Rtp1_inv = arma::inv(Ct.slice(t) + I_d);
    //Rtp1_inv = I_d;
    Bt = Ct.slice(t) * Rtp1_inv;
    //mT.col(t) = mt.col(t) + Bt*(mT.col(t+1) - mt.col(t));
    beta_nc.row(t) = arma::trans(mt.col(t) + Bt*(arma::trans(beta_nc.row(t+1)) - mt.col(t)));
    CT.slice(t) = St_tmp(N)/St_tmp(t)*(Ct.slice(t) - Bt*(Ct.slice(t) + I_d - St_tmp(t+1)/St_tmp(N)*CT.slice(t+1))*arma::trans(Bt));
    CT.slice(t) = 0.5*CT.slice(t) + 0.5*arma::trans(CT.slice(t));
    arma::mat tmp = Ct.slice(t) + arma::trans(beta_nc.row(t))*beta_nc.row(t);
    beta2_nc.row(t) = arma::trans(tmp.diag());
    tmp.diag().zeros();
    beta_nc_cov.row(t) = tmp;
    //beta_nc_samp.col(t) = mT.col(t);
  }

  //std::for_each(y.begin(), y.end(), res_protector);

  // for(int t = 0; t < N; t++){
  //   if(std::isnan(1/St(t))){
  //     Rcout << "St" << t <<": " << St(t) << "\n";
  //     break;
  //   }
  // }
  //std::for_each(beta_nc.begin(), beta_nc.end(), res_protector);
  //std::for_each(beta2_nc.begin(), beta2_nc.end(), res_protector);
  //std::for_each(beta_nc_cov.begin(), beta_nc_cov.end(), res_protector);
}

// [Rcpp::depends(RcppArmadillo)]
void update_prediction_error(arma::vec& y, arma::mat& x, arma::mat& beta_nc,
                             const arma::vec& theta_sr,
                             const arma::vec& beta_mean, const int N){
  arma::vec yt_star = y - x*beta_mean;
  arma::mat theta_sr_diag = arma::diagmat(theta_sr);
  arma::mat Ft = x*theta_sr_diag;
  for(int t=1; t < N+1; t++){
    y(t-1) = arma::as_scalar(yt_star(t-1) - Ft.row(t-1)*arma::trans(beta_nc.row(t)));
  }
}
