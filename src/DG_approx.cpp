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
  double proposal_mean = 0.3;
  double proposal_sd = 0.5;
  for(int i = 0; i < sample_size; i++){
    proposal(i) = R::rlnorm(proposal_mean, proposal_sd);
    //Rcout << "proposal: " << proposal(i) << "\n";
    //proposal(i) = R::rexp(1/b);
    //proposal(i) = R::runif(0.0, 1.0);
    part1(i) = d*std::log(proposal(i)) + d * scale_par_log - d*std::log(2.0) + param_vec_log_sum - 0.5 * scale_par * param_vec_sum - b;
    part2(i) = std::log(boost::math::tgamma(proposal(i)));
    r_likl(i) = proposal(i)*(part1(i)) - d*part2(i);
    p_likl(i) = R::dlnorm(proposal(i), proposal_mean, proposal_sd, true);
    //p_likl(i) = R::dexp(proposal(i), 1/b, true);
    //p_likl(i) = R::dunif(proposal(i), 0.0, 1.0, true);
    weights(i) = std::exp(r_likl(i) - p_likl(i));
  }
  if(!weights.is_finite()){
    index = arma::find_nonfinite(weights);
    Rcout << "part1: " << part1(index(0)) << "\n";
    Rcout << "part2: " << part2(index(0)) << "\n";
    Rcout << "part3: " << param_vec_log_sum << "\n";
    Rcout << "part4: " <<  0.5 * scale_par  << "\n";
    Rcout << "part5: " << param_vec_sum << "\n";

    stop("error: weights have non-finite elements.");
  }
  //Rcout << "sum weights: " << arma::sum(weights) << "\n";
  //Rcout << "r_likl: " << r_likl << "\n";
  //Rcout << "sum proposal weights: " << arma::sum(proposal % weights) << "\n";
  //Rcout << "expect: " << arma::sum(proposal % weights)/arma::sum(weights) << "\n";
  double result = arma::sum(proposal % weights)/arma::sum(weights);
  if(!arma::is_finite(result)){
    Rcout << "first part: " << arma::sum(proposal % weights) << "\n";
    Rcout << "second part: " << arma::sum(weights) << "\n";
    Rcout << "part3: " << param_vec_log_sum << "\n";
    Rcout << "part4: " <<  0.5 * scale_par  << "\n";
    Rcout << "part5: " << param_vec_sum << "\n";
    Rcout << "part6: " << arma::mean(r_likl - p_likl) << "\n";
    Rcout << "part7: " << arma::mean(proposal) << "\n";
    Rcout << "part8: " << arma::mean(r_likl) << "\n";
    Rcout << "part9: " << arma::mean(p_likl) << "\n";
    Rcout << "part10: " << arma::mean(part1) << "\n";
    Rcout << "part11: " << arma::mean(part2) << "\n";
  }
  return result;
}
