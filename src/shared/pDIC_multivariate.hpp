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

Rcpp::List compute_DIC(Rcpp::List temp_filter, arma::mat yt, arma::mat F1, arma::rowvec delta,
                       int sample_size, int i, int P, int chains=1){
    double ll = 0.0;
    double effective_size_mean = 0.0;
    // retrieve the values
    arma::mat mt = temp_filter["mt"];
    Rcpp::List Ct = temp_filter["Ct"];
    //arma::cube St = temp_filter["St"];
    arma::mat St = temp_filter["St"];
    int sign = 1;
    arma::mat F1t;
    //arma::mat Qt_tilde;
    double ll_mean = 0.0;
    int n_t = mt.n_cols;
    arma::dmat delta_m = arma::diagmat(arma::pow(delta, -0.5));
    arma::mat Qt;
    arma::vec ft;
    for(int j = P; j < (n_t - P); j++){
        F1t = arma::trans(gen_Ft(F1.col(j - sign * i)));
        ft = F1t * mt.col(j-1);
        //Qt = F1t * delta_m * Rcpp::as<arma::mat>(Ct(j-1)) * delta_m * arma::trans(F1t) + St.slice(j-1);
        Qt = F1t * delta_m * Rcpp::as<arma::mat>(Ct(j-1)) * delta_m * arma::trans(F1t) + St;
        arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(j)), ft, Qt, true);
        ll += arma::sum(tmp_ll);
        arma::mat Ct_tmp = Rcpp::as<arma::mat>(Ct(j));
        arma::mat sample_at = rmvnorm(sample_size, mt.col(j-1), Ct_tmp);
        arma::mat sample_ft = F1t * arma::trans(sample_at);
        arma::mat ll_sim(sample_size/chains, chains, arma::fill::zeros);
       #pragma omp parallel for num_threads(chains)
            for (int chain = 0; chain < chains; ++chain) {
                for(int k = 0; k < sample_size/chains; k++){
                    arma::vec tmp_ll_sim = dmvnorm(arma::trans(yt.col(j)), sample_ft.col(k), Qt, true);
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
