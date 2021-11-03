//
//  comp_spec.cpp
//  compute the spectral density matrix only
//
//  Created by Wenjie Zhao on 12/11/18.
//


#include <RcppArmadillo.h>
#include <Rcpp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List cp_sd(arma::cube phi, arma::cube SIGMA, arma::vec w){
    // phi is ar coefficient in array type;
       // SIGMA is innovation variance in matrix type;
       // w is the frequency band in vector type;
       // ch1 and ch2 is the index number of time series;
    int n_I = SIGMA.n_cols; // n_I is the number of time series;
    int P = phi.n_slices; // P is the optimizied model order;
    int n_t = phi.n_cols; // the number of time points;
    int n_w = w.n_elem; // the number of frequency points;

    // the variable storing all the information
    Rcpp::List sd1(n_t); // store the spectral density w/w.o time
    Rcpp::List sd2(n_t); // store partial directed coherence w/w.o time
    Rcpp::List sd3(n_t); // store directed transfer function
    arma::cx_cube f_spec(n_I, n_I, n_w, arma::fill::zeros);
    arma::cx_cube DTF_spec(n_I, n_I, n_w, arma::fill::zeros);
    arma::cx_cube kappa_spec(n_I, n_I, n_w, arma::fill::zeros);

    // temperaroy variable;
    arma::cx_mat PHI(n_I, n_I, arma::fill::eye);
    arma::vec temp;
    arma::mat temp_phi;
    arma::cx_mat PHI_inv(n_I, n_I);
    arma::cx_mat PHI_conj_inv(n_I, n_I);



    // some constants
    std::complex<double> ii(0, -2*M_PI);
    std::complex<double> exp_part;

    for(int i = 0; i < n_t; i++){
        for(int j = 0; j < n_w; j++){
            PHI.eye();
            for(int k = 0; k < P; k++){
                exp_part = std::exp(std::operator*(ii, (k+1)*w(j)));
                temp = phi(arma::span::all, arma::span(i), arma::span(k));
                temp_phi = arma::conv_to<arma::mat>::from(temp);
                temp_phi.reshape(n_I, n_I);
                PHI = PHI - exp_part*arma::trans(temp_phi);
            }
            PHI_inv = arma::inv(PHI);
            PHI_conj_inv = arma::inv(arma::trans(PHI));
            f_spec.slice(j) = PHI_inv * SIGMA.slice(i) * PHI_conj_inv;
            arma::cx_mat PHI_norm_tmp = arma::trans(PHI) * PHI;
            arma::cx_rowvec PHI_norm = arma::trans(arma::sqrt(PHI_norm_tmp.diag()));
            //partial directed coherence
            kappa_spec.slice(j) = PHI.each_row() / PHI_norm;

            // compute directed transfer function
            arma::cx_mat PHI_inv_norm_tmp = PHI_inv * arma::trans(PHI_inv);
            arma::cx_colvec PHI_inv_norm = arma::sqrt(PHI_inv_norm_tmp.diag());
            DTF_spec.slice(j) = PHI_inv.each_col() / PHI_inv_norm;
        }
        sd1(i) = f_spec;
        sd2(i) = kappa_spec;
        sd3(i) = DTF_spec;
    }
    return Rcpp::List::create(Rcpp::Named("sd") = sd1, Rcpp::Named("PDC") = sd2,
                              Rcpp::Named("DTF") = sd3);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat get_sd(Rcpp::List sd, int ts1, int ts2, int type){
    int n_t = sd.length();
    int n_w = Rcpp::as<arma::cx_cube>(sd(0)).n_slices;
    int n_I = Rcpp::as<arma::cx_cube>(sd(0)).n_rows;
    arma::cx_cube f_sd(n_I, n_I, n_w, arma::fill::zeros);
    arma::mat f_spec(n_I, n_I);
    arma::mat g_spec(n_I, n_I);
    arma::mat sd_tf(n_t, n_w, arma::fill::zeros);

    // Compute log spectral density of time series 1
    if(type == 1){
        for(int i = 0; i < n_t; i++){
            f_sd = Rcpp::as<arma::cx_cube>(sd(i));
            for(int j = 0; j < n_w; j++){
                f_spec = abs(f_sd.slice(j));
                sd_tf(i, j) = log(f_spec(ts1-1, ts1-1));
            }
        }
    }

    //
    if(type == 2){
        for(int i = 0; i < n_t; i++){
            f_sd = Rcpp::as<arma::cx_cube>(sd(i));
            for(int j = 0; j < n_w; j++){
                f_spec = abs(f_sd.slice(j));
                sd_tf(i, j) = (f_spec(ts1-1, ts2-1)*f_spec(ts1-1, ts2-1))/(f_spec(ts1-1, ts1-1)*f_spec(ts2-1, ts2-1));
            }
        }
    }

    if(type == 3){
        for(int i = 0; i < n_t; i++){
            f_sd = Rcpp::as<arma::cx_cube>(sd(i));
            for(int j = 0; j < n_w; j++){
                g_spec = abs(inv(f_sd.slice(j)));
                sd_tf(i, j) = (g_spec(ts1-1, ts2-1)*g_spec(ts1-1, ts2-1))/(g_spec(ts1-1, ts1-1)*g_spec(ts2-1, ts2-1));
            }
        }
    }

    if(type == 4){
        for(int i = 0; i < n_t; i++){
            f_sd = Rcpp::as<arma::cx_cube>(sd(i));
            for(int j = 0; j < n_w; j++){
                f_spec = abs(f_sd.slice(j));
                sd_tf(i, j) = f_spec(ts1-1, ts2-1);
            }
        }
    }
    return(sd_tf);
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List cp_sd_nt(arma::cube phi, arma::mat SIGMA, arma::vec w){
    // phi is ar coefficient in array type;
    // SIGMA is innovation variance in matrix type;
    // w is the frequency band in vector type;
    // ch1 and ch2 is the index number of time series;
    int n_I = SIGMA.n_cols; // n_I is the number of time series;
    int P = phi.n_slices; // P is the optimizied model order;
    int n_w = w.n_elem; // the number of frequency points;

    // the variable storing all the information
    arma::cx_cube f_spec(n_I, n_I, n_w, arma::fill::zeros);
    arma::cx_cube g_spec(n_I, n_I, n_w, arma::fill::zeros);
    arma::cx_cube DTF_spec(n_I, n_I, n_w, arma::fill::zeros);
    arma::cx_cube kappa_spec(n_I, n_I, n_w, arma::fill::zeros);

    // temperaroy variable;
    arma::cx_mat PHI(n_I, n_I, arma::fill::eye);
    arma::cx_mat PHI_inv(n_I, n_I);
    arma::cx_mat PHI_conj_inv(n_I, n_I);

    // some constants
    std::complex<double> ii(0, -2*M_PI);
    std::complex<double> exp_part;

    for(int j = 0; j < n_w; j++){
        PHI.eye();
        for(int k = 0; k < P; k++){
            exp_part = std::exp(std::operator*(ii, (k+1)*w(j)));
            PHI = PHI - exp_part*arma::trans(phi.slice(k));
        }
        PHI_inv = arma::inv(PHI);
        PHI_conj_inv = arma::inv(arma::trans(PHI));
        f_spec.slice(j) = PHI_inv * SIGMA * PHI_conj_inv;
        g_spec.slice(j) = arma::inv(f_spec.slice(j));

        arma::cx_mat PHI_norm_tmp = arma::trans(PHI) * PHI;
        arma::cx_rowvec PHI_norm = arma::trans(arma::sqrt(PHI_norm_tmp.diag()));

        // partial directed coherence
        kappa_spec.slice(j) = PHI.each_row() / PHI_norm;

        // compute directed transfer function
        arma::cx_mat PHI_inv_norm_tmp = PHI_inv * arma::trans(PHI_inv);
        arma::cx_colvec PHI_inv_norm = arma::sqrt(PHI_inv_norm_tmp.diag());
        DTF_spec.slice(j) = PHI_inv.each_col() / PHI_inv_norm;
    }
    return Rcpp::List::create(Rcpp::Named("f_spec") = f_spec,
                              Rcpp::Named("g_spec") = g_spec,
                              Rcpp::Named("kappa_spec") = kappa_spec,
                              Rcpp::Named("DTF_spec") = DTF_spec);
}




// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat get_sd_nt(Rcpp::List sd, int ts1, int ts2){
    int n_w = Rcpp::as<arma::cx_cube>(sd["f_spec"]).n_slices;
    int n_I = Rcpp::as<arma::cx_cube>(sd["f_spec"]).n_rows;
    arma::cx_cube f_sd(n_I, n_I, n_w, arma::fill::zeros);
    arma::mat f_spec(n_I, n_I);
    arma::mat g_spec(n_I, n_I);
    arma::mat kappa_spec(n_I, n_I);
    arma::mat DTF_spec(n_I, n_I);
    arma::mat sd_tf(6, n_w, arma::fill::zeros);

    // Compute log spectral density of time series 1
    f_sd = Rcpp::as<arma::cx_cube>(sd["f_spec"]);
    for(int j = 0; j < n_w; j++){
        f_spec = abs(f_sd.slice(j));
        sd_tf(0, j) = log(f_spec(ts1-1, ts1-1));
    }

    // Compute log spectral density of time series 2
    f_sd = Rcpp::as<arma::cx_cube>(sd["f_spec"]);
    for(int j = 0; j < n_w; j++){
        f_spec = abs(f_sd.slice(j));
        sd_tf(1, j) = log(f_spec(ts2-1, ts2-1));
    }

    // Compute coherence between time series 1 and time series 2
    f_sd = Rcpp::as<arma::cx_cube>(sd["f_spec"]);
    for(int j = 0; j < n_w; j++){
        f_spec = abs(f_sd.slice(j));
        sd_tf(2, j) = (f_spec(ts1-1, ts2-1)*f_spec(ts1-1, ts2-1))/(f_spec(ts1-1, ts1-1)*f_spec(ts2-1, ts2-1));
    }

    // Compute partial coherence between time series 1 and time series 2
    f_sd = Rcpp::as<arma::cx_cube>(sd["g_spec"]);
    for(int j = 0; j < n_w; j++){
        g_spec = abs(f_sd.slice(j));
        sd_tf(3, j) = (g_spec(ts1-1, ts2-1)*g_spec(ts1-1, ts2-1))/(g_spec(ts1-1, ts1-1)*g_spec(ts2-1, ts2-1));
    }

    // Compute directed partial coherence from ts1 to ts2
    f_sd = Rcpp::as<arma::cx_cube>(sd["kappa_spec"]);
    for(int j = 0; j < n_w; j++){
        kappa_spec = abs(f_sd.slice(j));
        sd_tf(4, j) = kappa_spec(ts1-1, ts2-1);
    }


    f_sd = Rcpp::as<arma::cx_cube>(sd["DTF_spec"]);
    for(int j = 0; j < n_w; j++){
        DTF_spec = abs(f_sd.slice(j));
        sd_tf(5, j) = DTF_spec(ts1-1, ts2-1);
    }
    return(sd_tf);
}
