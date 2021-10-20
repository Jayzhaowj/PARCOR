#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <math.h>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// Wrap up bessel function
double bessel_k(double nu, double xx){
  return boost::math::cyl_bessel_k(nu, xx);
}

// [[Rcpp::depends(RcppArmadillo)]]
double Egig_log(double p, double a, double b){
  double alpha_bar = std::sqrt(a*b);
  Environment numDeriv("package:numDeriv");
  Function grad = numDeriv["grad"];
  NumericVector Kderiv = grad(Rcpp::_["func"] = Rcpp::InternalFunction(bessel_k),
                              Rcpp::_["xx"] = alpha_bar,
                              Rcpp::_["x"] = p,
                              Rcpp::_["method.args"] = Rcpp::List::create(Rcpp::_["eps"] = 1e-8, Rcpp::_["show.details"] = false));
  double result = 0.5*(std::log(b) - std::log(a)) + Kderiv[0]/boost::math::cyl_bessel_k(p, alpha_bar);
  return result;
}


