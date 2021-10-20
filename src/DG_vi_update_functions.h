#ifndef DG_SAMPLING_FUNCTIONS_H
#define DG_SAMPLING_FUNCTIONS_H

void update_local_shrink(arma::vec& local_shrink,
                         arma::vec& local_shrink_inv,
                         arma::vec& local_shrink_log,
                         const arma::vec& param_vec2,
                         double global_shrink,
                         double a);

void update_global_shrink(const arma::vec& prior_var,
                          double& global_shrink,
                          double& global_shrink_log,
                          double a,
                          double hyper1,
                          double hyper2);

#endif
