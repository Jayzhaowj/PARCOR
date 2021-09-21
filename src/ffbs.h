#ifndef FFBS_H
#define FFBS_H


Rcpp::List forward_filter_backward_smooth(arma::mat yt, arma::mat F1, arma::mat F2,
                                          double n_0, double S_0,
                                          int n_t, int n_I, int m, int type, int P,
                                          double delta1, double delta2, int sample_size);


Rcpp::List sample_parcor_hier(Rcpp::List result, int m, int P, int type,
                              int sample_size);

void update_beta_tilde(arma::mat& beta_nc,
                       arma::mat& beta2_nc,
                       arma::cube& beta_nc_cov,
                       arma::vec& y, arma::mat& x,
                       const arma::vec& theta_sr,
                       const arma::vec& beta_mean, const int N,
                       const double S_0,
                       arma::vec& St);

Rcpp::List ffbs_DIC(arma::mat yt, arma::mat F1, arma::mat F2,
                    double n_0, double S_0,
                    int n_t, int n_I, int m, int type, int P,
                    arma::mat delta, bool DIC, int sample_size,
                    int chains, bool uncertainty);

void update_prediction_error(arma::vec& y, arma::mat& x, arma::mat& beta_nc,
                                  const arma::vec& theta_sr,
                                  const arma::vec& beta_mean, const int N);
#endif
