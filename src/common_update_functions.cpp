#include <RcppArmadillo.h>
#include <math.h>
#include "sample_parameters.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
void update_beta_mean(arma::vec& beta_mean,
                      arma::vec& beta2_mean,
                      arma::vec& theta_sr,
                      const arma::vec& y,
                      const arma::mat& x,
                      const arma::mat& beta_nc,
                      const arma::vec& sigma2_inv,
                      const arma::vec& tau2_inv){
  // beta_nc dimension: N * d
  int d = x.n_cols;
  arma::mat x_tilde = x % beta_nc;
  arma::vec sigma2_beta_mean(d);
  arma::mat part1 = x_tilde * theta_sr;
  arma::mat part2 = x * beta_mean;
  arma::mat tmp;
  for(int i = 0; i < d; i++){
    sigma2_beta_mean(i) = 1.0/arma::as_scalar(arma::sum(sigma2_inv%arma::square(x.col(i))) + tau2_inv(i));
    tmp = (y - part2 - part1 + x.col(i) * beta_mean(i));
    beta_mean(i) = arma::as_scalar(sigma2_beta_mean(i) * (arma::sum(sigma2_inv % (tmp % x.col(i)))));
    beta2_mean(i) = arma::as_scalar(sigma2_beta_mean(i)) + arma::as_scalar(beta_mean(i)*beta_mean(i));
  }
  //std::for_each(beta_mean.begin(), beta_mean.end(), res_protector);
  //std::for_each(beta2_mean.begin(), beta2_mean.end(), res_protector);
}

// [[Rcpp::depends(RcppArmadillo)]]
void update_theta_sr(arma::vec& beta_mean,
                     arma::vec& theta_sr,
                     arma::vec& theta,
                     const arma::vec& y,
                     const arma::mat& x,
                     const arma::mat& beta_nc,
                     const arma::mat& beta2_nc,
                     const arma::cube& beta_cov_nc,
                     const arma::vec& sigma2_inv,
                     const arma::vec& xi2_inv){
  // beta_nc dimension: N * d
  // beta_cov_nc dimension: N*d*d
  int d = x.n_cols;

  arma::mat x2 = arma::pow(x, 2);
  arma::mat x2_tilde = x2 % beta2_nc;
  arma::vec sigma2_theta_sr_mean(d);
  arma::mat part2 = x * beta_mean;
  arma::mat part1;
  arma::mat tmp;
  for(int i = 0; i < d; i++){
    sigma2_theta_sr_mean(i) = 1.0/arma::as_scalar(arma::sum(sigma2_inv%x2_tilde.col(i)) + xi2_inv(i));
    part1 = x % beta_cov_nc.slice(i) * theta_sr;
    tmp = y % beta_nc.col(i) - beta_nc.col(i) % part2 - part1;
    theta_sr(i) = arma::as_scalar(sigma2_theta_sr_mean(i) * (arma::sum(sigma2_inv % (tmp % x.col(i)))));
    theta(i) = arma::as_scalar(sigma2_theta_sr_mean(i)) + arma::as_scalar(theta_sr(i)*theta_sr(i));
  }
  //std::for_each(theta_sr.begin(), theta_sr.end(), res_protector);
  //std::for_each(theta.begin(), theta.end(), res_protector);
}




// [[Rcpp::depends(RcppArmadillo)]]
void update_beta(arma::vec& beta,
                 arma::vec& beta2,
                 arma::vec& sigma2_beta,
                 const arma::vec& y,
                 const arma::mat& x,
                 const double& sigma2_inv,
                 const arma::vec& tau2_inv){
  // beta_nc dimension: N * d
  int d = x.n_cols;
  int n_t = x.n_rows;
  arma::vec sigma2t_inv(n_t, arma::fill::ones);
  sigma2t_inv.fill(sigma2_inv);
  arma::mat part2 = x * beta;
  arma::mat tmp;
  for(int i = 0; i < d; i++){
    sigma2_beta(i) = 1.0/arma::as_scalar(arma::sum(sigma2t_inv%arma::square(x.col(i))) + tau2_inv(i));
    tmp = y - part2 + x.col(i) * beta(i);
    beta(i) = arma::as_scalar(sigma2_beta(i) * (arma::sum(sigma2t_inv % (tmp % x.col(i)))));
    beta2(i) = arma::as_scalar(sigma2_beta(i)) + arma::as_scalar(beta(i)*beta(i));
  }
  //std::for_each(beta_mean.begin(), beta_mean.end(), res_protector);
  //std::for_each(beta2_mean.begin(), beta2_mean.end(), res_protector);
}

//[[Rcpp::depends(RcppArmadillo)]]
void update_sigma2(arma::vec& beta, arma::vec& sigma2_beta,
                   arma::vec& y, arma::mat& x, double& sigma2, double& sigma2_inv,
                   const double& c0, const double& C0){
  int n_t = y.n_elem;
  double par_alpha = n_t/2.0 + c0;
  double par_beta;
  double tmp = 0.0;
  arma::vec y2 = arma::square(y);
  arma::mat x2 = arma::square(x);
  for(int i = 0; i < n_t; i++){
    tmp += y2(i) + arma::as_scalar(x2.row(i)*sigma2_beta) + arma::as_scalar(arma::square(x.row(i)*beta)) - 2*y(i)*arma::as_scalar(x.row(i)*beta);
  }
  par_beta = tmp/2.0 + C0;
  sigma2 = par_beta/(par_alpha + 1);
  sigma2_inv = par_alpha/par_beta;
}

//[[Rcpp::depends(RcppArmadillo)]]
void update_C0(double& C0, const double& sigma2_inv, const double& c0,
               const double& g0, const double& G0){
  double alpha = c0 + g0;
  double beta = G0 + sigma2_inv;
  C0 = alpha/beta;
}
