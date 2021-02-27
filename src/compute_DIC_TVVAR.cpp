//
//  compute_DIC_TVVAR.cpp
//
//
//  Created by Wenjie Zhao on 1/16/19.
//

#ifndef __compute_DIC_TVVAR__
#define __compute_DIC_TVVAR__


#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <RcppDist.h>

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]

Rcpp::List compute_DIC_TVVAR(Rcpp::List temp_filter, int sample_size, arma::cube St, int P_max){
    double effective_size_mean = 0.0;
    double DIC = 0.0;
    double ll = 0.0;
    // retrieve the values
    arma::mat at = temp_filter["at"];
    arma::mat mt = temp_filter["mt"];
    arma::cube Ct = temp_filter["Ct"];
    arma::cube F1t = temp_filter["F1t"];
    arma::mat ft = temp_filter["ft"];
    arma::mat yt = temp_filter["yt"];
    arma::cube Qt = temp_filter["Qt"];
    double ll_mean = 0.0;
    int n_t = at.n_cols;
    for(int j = P_max; j < (n_t - P_max); j++){
        arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(j)), ft.col(j), St.slice(j), true);
        ll += arma::sum(tmp_ll);
        arma::mat sample_at = rmvnorm(sample_size, at.col(j), Ct.slice(j));
        arma::mat sample_ft = F1t.slice(j) * arma::trans(sample_at);
        arma::vec ll_sim(sample_size, arma::fill::zeros);
        for(int k = 0; k < sample_size; k++){
            arma::vec tmp_ll_sim = dmvnorm(arma::trans(yt.col(j)), sample_ft.col(k), St.slice(j), true);
            ll_sim(k) = arma::sum(tmp_ll_sim);
        }
        ll_mean += arma::mean(ll_sim);
    }
    effective_size_mean = 2*(ll - ll_mean);
    DIC = 2*(effective_size_mean - ll);
    return Rcpp::List::create(Rcpp::Named("DIC") = DIC, Rcpp::Named("ll") = ll, Rcpp::Named("ES_mean") = effective_size_mean, Rcpp::Named("ll_mean") = ll_mean);
}

#endif /* compute_DIC_TVVAR */

