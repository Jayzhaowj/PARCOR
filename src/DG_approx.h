#ifndef DG_APPROX_H
#define DG_APPROX_H

double DG_approx(const arma::vec& param_vec,
                 const arma::vec& param_vec_log,
                 double scale_par,
                 double scale_par_log,
                 double b,
                 int sample_size);
#endif
