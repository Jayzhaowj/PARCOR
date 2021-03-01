// compute spectral density for dynamic hierarchical model

#include <RcppArmadillo.h>
#include <Rcpp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::cube cp_sd_uni(arma::cube phi, arma::vec sigma2,
                    arma::vec w){
  int n_t = phi.n_cols;
  int n_I = phi.n_rows;
  int P = phi.n_slices;
  int n_w = w.n_elem;
  arma::mat sd_tf(n_t, n_w, arma::fill::zeros);
  arma::cube f_spec(n_I, n_t, n_w, arma::fill::zeros);
  arma::cx_vec exp_part(P);
  arma::vec lag = arma::linspace(1, P, P);
  // some constants
  std::complex<double> ii(0, -2*M_PI);
  //std::complex<double> exp_part;

  for(int i = 0; i < n_t; i++){
    for(int j = 0; j < n_w; j++){
      exp_part = exp(ii*lag*w(j));
      for(int k = 0; k < n_I; k++){
        f_spec(k, i, j) = log(sigma2(i)) - 2*log(abs(1.0-sum(arma::vectorise(phi(arma::span(k),arma::span(i), arma::span::all)) % exp_part)));
      }
    }
  }
  return f_spec;
}

