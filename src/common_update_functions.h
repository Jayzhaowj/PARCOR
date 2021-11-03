
#ifndef COMMON_UPDATE_FUNCTIONS_H
#define COMMON_UPDATE_FUNCTIONS_H

void update_beta_mean(arma::vec& beta_mean,
                      arma::vec& beta2_mean,
                      arma::vec& theta_sr,
                      const arma::vec& y,
                      const arma::mat& x,
                      const arma::mat& beta_nc,
                      const arma::vec& sigma2_inv,
                      const arma::vec& tau2_inv);

void update_theta_sr(arma::vec& beta_mean,
                     arma::vec& theta_sr,
                     arma::vec& theta,
                     const arma::vec& y,
                     const arma::mat& x,
                     const arma::mat& beta_nc,
                     const arma::mat& beta2_nc,
                     const arma::cube& beta_cov_nc,
                     const arma::vec& sigma2_inv,
                     const arma::vec& xi2_inv);

void update_beta(arma::vec& beta,
                 arma::vec& beta2,
                 arma::vec& sigma2_beta,
                 const arma::vec& y,
                 const arma::mat& x,
                 const double& sigma2_inv,
                 const arma::vec& tau2_inv);


void update_sigma2(arma::vec& beta, arma::vec& sigma2_beta,
                   arma::vec& y, arma::mat& x, double& sigma2, double& sigma2_inv,
                   const double& c0, const double& C0);

void update_C0(double& C0, const double& sigma2_inv, const double& c0,
               const double& g0, const double& G0);

#endif
