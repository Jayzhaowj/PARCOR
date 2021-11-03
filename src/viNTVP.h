#ifndef VINTVP_CPP_H
#define VINTVP_CPP_H

List vi_shrinkNTVP(arma::mat y_fwd,
                   arma::mat y_bwd,
                   int d,
                   double e1,
                   double e2,
                   double c0,
                   double g0,
                   double G0,
                   double a_tau,
                   bool learn_a_tau,
                   bool learn_sigma2,
                   int iter_max,
                   bool ind,
                   double epsilon,
                   bool skip,
                   int sample_size,
                   double b_tau);

#endif
