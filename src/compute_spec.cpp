//
//  compute_spec.cpp
//  compute the spectral density, squared coherence and squared partial coherence of two time series
//
//  Created by Wenjie Zhao on 12/11/18.
//


#include <RcppArmadillo.h>
#include <Rcpp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List compute_spec(arma::cube phi, arma::mat SIGMA,
                        arma::vec w, int P_max, int ch1,
                        int ch2, bool time_depend = true){
  // phi is ar coefficient in array type;
  // SIGMA is innovation variance in matrix type;
  // w is the frequency band in vector type;
  // P_max is the potential model order;
  // ch1 and ch2 is the index number of time series;
  int n_I = SIGMA.n_cols; // n_I is the number of time series;
  int P = phi.n_slices; // P is the optimizied model order;
  int n_t = phi.n_cols; // the number of time points;
  int n_w = w.n_elem; // the number of frequency points;
  int index1 = ch1 - 1;
  int index2 = ch2 - 1;

  // temperaroy variable;
  arma::cx_mat PHI(n_I, n_I, arma::fill::eye);
  arma::vec temp;
  arma::mat temp_phi;
  arma::cx_mat PHI_inv(n_I, n_I);
  arma::cx_mat PHI_conj_inv(n_I, n_I);
  arma::mat SIGMA_inv = arma::inv(SIGMA);
  // f_spec and g_spec are spectral density and precision matrix.
  // f_dens and g_dens is the modulus of f_spec and g_spec;
  arma::cx_mat f_spec(n_I, n_I);
  arma::cx_mat g_spec(n_I, n_I);
  arma::mat f_dens(n_I, n_I);
  arma::mat g_dens(n_I, n_I);
  arma::mat kappa_dens(n_I, n_I);
  arma::mat DTF_dens(n_I, n_I);
  // some constants
  std::complex<double> ii(0, -2*M_PI);
  std::complex<double> exp_part;
  Rcpp::List sd(6); //spectral density of the first process and the second process, squared coherence and partial squared coherence
  if(time_depend){
    arma::mat sd1(n_t, n_w, arma::fill::zeros);
    arma::mat sd2(n_t, n_w, arma::fill::zeros);
    arma::mat sd3(n_t, n_w, arma::fill::zeros);
    arma::mat sd4(n_t, n_w, arma::fill::zeros);
    arma::mat sd5(n_t, n_w, arma::fill::zeros);
    arma::mat sd6(n_t, n_w, arma::fill::zeros);
    for(int i = P_max; i < n_t - P_max; i++){
      for(int j = 0; j < n_w; j++){
        PHI.eye();
        for(int k = 0; k < P; k++){
          exp_part = std::exp(std::operator*(ii, (k+1)*w(j)));
          temp = phi(arma::span::all, arma::span(i), arma::span(k));
          temp_phi = arma::conv_to<arma::mat>::from(temp);
          temp_phi.reshape(n_I, n_I);
          PHI = PHI - exp_part*arma::trans(temp_phi);
        }
        PHI_inv = arma::inv(PHI);
        PHI_conj_inv = arma::inv(arma::trans(PHI));
        arma::cx_mat PHI_norm_tmp = arma::trans(PHI) * PHI;
        arma::cx_rowvec PHI_norm = arma::trans(arma::sqrt(PHI_norm_tmp.diag()));
        arma::cx_mat PI_val = PHI.each_row() / PHI_norm;
        //arma::cx_mat kappa = arma::trans(PI_val) * SIGMA_inv * PI_val;


        // compute directed transfer function
        arma::cx_mat PHI_inv_norm_tmp = PHI_inv * arma::trans(PHI_inv);
        arma::cx_colvec PHI_inv_norm = arma::sqrt(PHI_inv_norm_tmp.diag());
        arma::cx_mat DTF = PHI_inv.each_col() / PHI_inv_norm;
        DTF_dens = abs(DTF); // directed transfer function


        f_spec = PHI_inv * SIGMA * PHI_conj_inv;
        g_spec = inv(f_spec);
        f_dens = abs(f_spec);
        g_dens = abs(g_spec);
        kappa_dens = abs(PI_val); // partial directed coherence


        sd1(i, j) = log(f_dens(index1, index1)); //spectral density of the first process
        sd2(i, j) = log(f_dens(index2, index2)); //spectral density of the second process
        sd3(i, j) = (f_dens(index1, index2)*f_dens(index1, index2))/(f_dens(index1, index1) * f_dens(index2, index2)); // squared coherence between two processes
        sd4(i, j) = (g_dens(index1, index2)*g_dens(index1, index2))/(g_dens(index1, index1) * g_dens(index2, index2)); // squared partial coherence between two processes
        sd5(i, j) = kappa_dens(index1, index2);
        sd6(i, j) = DTF_dens(index1, index2);
      }
    }
    //sd(1) = Rcpp::as<arma::mat>(sd1);
    //sd(2) = Rcpp::as<arma::mat>(sd2);
    //sd(3) = Rcpp::as<arma::mat>(sd3);
    //sd(4) = Rcpp::as<arma::mat>(sd4);
    sd(0) = sd1;
    sd(1) = sd2;
    sd(2) = sd3;
    sd(3) = sd4;
    sd(4) = sd5;
    sd(5) = sd6;
  }else{
    arma::vec sd1(n_w);
    arma::vec sd2(n_w);
    arma::vec sd3(n_w);
    arma::vec sd4(n_w);
    arma::vec sd5(n_w);
    arma::vec sd6(n_w);
    for(int j = 0; j < n_w; j++){
      PHI.eye();
      for(int k = 0; k < P; k++){
        exp_part = std::exp(std::operator*(ii, (k+1)*w(j)));
        temp = phi(arma::span::all, arma::span(P_max), arma::span(k));
        temp_phi = arma::conv_to<arma::mat>::from(temp);
        temp_phi.reshape(n_I, n_I);
        PHI = PHI - exp_part*temp_phi;
      }
      PHI_inv = arma::inv(PHI);
      PHI_conj_inv = arma::inv(arma::trans(PHI));
      arma::cx_mat PHI_norm_tmp = arma::trans(PHI) * PHI;
      arma::cx_rowvec PHI_norm = arma::trans(arma::sqrt(PHI_norm_tmp.diag()));
      arma::cx_mat PI_val = PHI.each_row() / PHI_norm;
      //arma::cx_mat kappa = arma::trans(PI_val) * SIGMA_inv * PI_val;
      kappa_dens = abs(PI_val); // partial directed coherence


      // compute directed transfer function
      arma::cx_mat PHI_inv_norm_tmp = PHI_inv * arma::trans(PHI_inv);
      arma::cx_colvec PHI_inv_norm = arma::sqrt(PHI_inv_norm_tmp.diag());
      arma::cx_mat DTF = PHI_inv.each_col() / PHI_inv_norm;
      DTF_dens = abs(DTF); // directed transfer function

      f_spec = PHI_inv * SIGMA * PHI_conj_inv;
      g_spec = inv(f_spec);
      f_dens = abs(f_spec);
      g_dens = abs(g_spec);
      sd1(j) = log(f_dens(index1, index1)); //spectral density of the first process
      sd2(j) = log(f_dens(index2, index2)); //spectral density of the second process
      sd3(j) = (f_dens(index1, index2)*f_dens(index1, index2))/(f_dens(index1, index1) * f_dens(index2, index2)); // squared coherence between two processes
      sd4(j) = (g_dens(index1, index2)*g_dens(index1, index2))/(g_dens(index1, index1) * g_dens(index2, index2)); // squared partial coherence between two processes
      sd5(j) = kappa_dens(index1, index2);
      sd6(j) = DTF_dens(index1, index2);
    }
    //sd(1) = Rcpp::as<arma::vec>(sd1);
    //sd(2) = Rcpp::as<arma::vec>(sd2);
    //sd(3) = Rcpp::as<arma::vec>(sd3);
    //sd(4) = Rcpp::as<arma::vec>(sd4);
    sd(0) = sd1;
    sd(1) = sd2;
    sd(2) = sd3;
    sd(3) = sd4;
    sd(4) = sd5;
    sd(5) = sd6;
  }
  return(sd);
}
