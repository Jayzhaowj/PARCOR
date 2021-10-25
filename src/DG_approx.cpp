#include <RcppArmadillo.h>
#include <boost/math/special_functions/gamma.hpp>
#include <Rmath.h>
using namespace Rcpp;

double DG_approx(const arma::vec& param_vec,
                 const arma::vec& param_vec_log,
                 double scale_par,
                 double scale_par_log,
                 double b,
                 int sample_size){
  int d = param_vec.n_elem;
  arma::vec r_likl(sample_size, arma::fill::zeros);
  arma::vec p_likl(sample_size, arma::fill::zeros);
  arma::vec weights(sample_size, arma::fill::zeros);
  arma::vec proposal(sample_size, arma::fill::zeros);
  double param_vec_sum = arma::sum(param_vec);
  double param_vec_log_sum = arma::sum(param_vec_log);
  arma::uvec index;
  arma::vec part1(sample_size, arma::fill::zeros);
  arma::vec part2(sample_size, arma::fill::zeros);

  for(int i = 0; i < sample_size; i++){
    proposal(i) = R::rexp(1/b);
    part1(i) = d*std::log(proposal(i)) + d * scale_par_log - d*std::log(2.0) + param_vec_log_sum - 0.5 * scale_par * param_vec_sum - b;
    part2(i) = std::log(boost::math::tgamma(proposal(i)));
    r_likl(i) = proposal(i)*(part1(i)) - d*part2(i);
    p_likl(i) = R::dexp(proposal(i), 1/b, true);
    weights(i) = std::exp(r_likl(i) - p_likl(i));
  }
  double result = arma::sum(proposal % weights)/arma::sum(weights);
  return result;
}
