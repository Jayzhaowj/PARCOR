//
//  DIC.hpp
//
//
//  Created by Wenjie Zhao on 10/24/19.
//

#ifndef DIC_hpp
#define DIC_hpp


#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "gen_F1t.hpp"
#include <omp.h>
#include <RcppDist.h>


// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]

Rcpp::List compute_DIC(Rcpp::List temp_filter, arma::mat F1,
                       int sample_size, int i, int P, int chains=1){
    double ll = 0.0;
    double effective_size_mean = 0.0;
    // retrieve the values
    arma::mat at = temp_filter["at"];
    arma::mat mt = temp_filter["mt"];
    Rcpp::List Ct = temp_filter["Ct"];
    arma::mat ft = temp_filter["ft"];
    arma::mat yt = temp_filter["yt"];
    arma::cube Qt = temp_filter["Qt"];
    int sign = 1;
    arma::mat F1t;
    //arma::mat Qt_tilde;
    double ll_mean = 0.0;
    int n_t = at.n_cols;

    for(int j = P; j < (n_t - P); j++){
        arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(j)), ft.col(j), Qt.slice(j), true);
        ll += arma::sum(tmp_ll);
        arma::mat Ct_tmp = Rcpp::as<arma::mat>(Ct(j));
        arma::mat sample_at = rmvnorm(sample_size, at.col(j), Ct_tmp);
        F1t = arma::trans(gen_Ft(F1.col(j - sign * i)));
        arma::mat sample_ft = F1t * arma::trans(sample_at);
        arma::mat ll_sim(sample_size/chains, chains, arma::fill::zeros);
       #pragma omp parallel for num_threads(chains)
            for (int chain = 0; chain < chains; ++chain) {
                for(int k = 0; k < sample_size/chains; k++){
                    arma::vec tmp_ll_sim = dmvnorm(arma::trans(yt.col(j)), sample_ft.col(k), Qt.slice(j), true);
                        //arma::vec tmp_ll_sim = dmvnorm(arma::trans(yt.col(j)), sample_ft.col(k), Qt_tilde, true);
                ll_sim(k, chain) = arma::sum(tmp_ll_sim);
            }
        }
        ll_mean += arma::mean(arma::vectorise(ll_sim));
    }
    effective_size_mean = 2*(ll - ll_mean);
    return Rcpp::List::create(Rcpp::Named("ll") = ll, Rcpp::Named("ES_mean") = effective_size_mean);
}


#endif /* DIC_hpp */
