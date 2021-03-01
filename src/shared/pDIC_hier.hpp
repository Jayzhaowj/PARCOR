//
//  pDIC.hpp
//
//
//  Created by Wenjie Zhao on 10/24/19.
//

#ifndef pDIC_hpp
#define pDIC_hpp


#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <omp.h>
#include <RcppDist.h>


// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::export]]

double compute_pDIC(Rcpp::List temp_filter, arma::mat yt, arma::mat F1,
                    arma::mat F2, int sample_size, int m, int P, int type,
                    int chains, double ll){
    //double ll = 0.0;
    double effective_size_mean = 0.0;
    // retrieve the values
    arma::mat akt = temp_filter["akt"];
    arma::cube Rkt = temp_filter["Rkt"];
    //arma::mat at = temp_filter["at"];
    arma::cube Rt = temp_filter["Rt"];
    //arma::mat ft = temp_filter["ft"];
    arma::cube Qt = temp_filter["Qt"];
    //arma::vec St = temp_filter["sigma2t"];
    //arma::vec nt = temp_filter["nt"];
    int sign = 1;
    if(type == 1){
        sign = 1;
    }else{
        sign = -1;
    }

    int n_I = akt.n_rows;
    int n_t = akt.n_cols;
    arma::cube F1t(n_I, n_I, n_t, arma::fill::zeros);
    //arma::mat Qt_tilde;
    double ll_mean = 0.0;
    for(int j = P; j < (n_t - P); j++){
        //Qt_tilde = F1t.slice(j)*Rt.slice(j)*arma::trans(F1t.slice(j)) + St.slice(j);
        //arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(j)), ft.col(j), Qt_tilde, true);
        F1t.slice(j) = arma::diagmat(F1.col(j - sign*m));
        //arma::vec tmp_ll = dmvnorm(arma::trans(yt.col(j)), ft.col(j), Qt.slice(j), true);
        //ll += arma::sum(tmp_ll);

        //arma::mat sample_akt = rmvnorm(sample_size, akt.col(j), Rkt.slice(j));
        //arma::mat sample_at = F2*arma::trans(sample_akt);
        //arma::mat sample_ft = F1t.slice(j) * sample_at;
        arma::mat ll_sim(sample_size/chains, chains, arma::fill::zeros);
       #pragma omp parallel for num_threads(chains)
            for (int chain = 0; chain < chains; ++chain) {
                for(int k = 0; k < sample_size/chains; k++){
                    //arma::sample_Qt = F1t.slice(j) *
                    arma::mat sample_akt = rmvnorm(1, akt.col(j), Rkt.slice(j));
                    arma::mat sample_at = rmvnorm(1, F2*arma::trans(sample_akt), Rt.slice(j));
                    arma::mat sample_ft = F1t.slice(j)*arma::trans(sample_at);
                    //arma::mat sample_ft = F1t.slice(j)*at.col(j);
                    arma::vec tmp_ll_sim = dmvnorm(arma::trans(yt.col(j)),
                                                   sample_ft, Qt.slice(j), true);
                    ll_sim(k, chain) = arma::sum(tmp_ll_sim);
            }
        }

        ll_mean += arma::mean(arma::vectorise(ll_sim));
    }
    effective_size_mean = 2*(ll - ll_mean);
    return effective_size_mean;
}


#endif /* DIC_hpp */
