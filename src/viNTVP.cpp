// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <progress.hpp>
#include <math.h>
#include "ffbs.h"
#include "DG_vi_update_functions.h"
#include "common_update_functions.h"
#include "sample_parameters.h"
#include "DG_approx.h"
#include "utilities_cpp.h"
using namespace Rcpp;

// [[Rcpp::export]]
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
                   int iter_max,
                   bool ind,
                   double epsilon,
                   int sample_size,
                   double b_tau) {

  // Progress bar setup
  arma::vec prog_rep_points = arma::round(arma::linspace(0, iter_max, 50));
  Progress p(50, true);

  // Some necessary dimensions
  int N = y_fwd.n_rows;
  int n_I = y_fwd.n_cols;

  // Some index
  //int m = 1; // current stage
  int N_m;
  int n_1;     // index
  int n_T;     // index

  int start = 1;


  // generate forward and backward prediction error
  arma::cube yf(N, n_I, d+1, arma::fill::none);
  arma::cube yb(N, n_I, d+1, arma::fill::none);
  yf.slice(start - 1) = y_fwd;
  yb.slice(start - 1) = y_bwd;

  arma::vec y_tmp;
  arma::mat x_tmp;
  int d_tmp;
  int counts;
  arma::vec diff((d-start+1)*12 + 16, arma::fill::zeros);

  // generate forward and backward PARCOR
  // Variance-covariance
  arma::mat sigma2f_old(n_I, d, arma::fill::none);
  arma::mat sigma2b_old(n_I, d, arma::fill::none);

  arma::mat sigma2f_new(n_I, d, arma::fill::none);
  arma::mat sigma2b_new(n_I, d, arma::fill::none);

  arma::mat sigma2f_inv_old(n_I, d, arma::fill::none);
  arma::mat sigma2b_inv_old(n_I, d, arma::fill::none);

  arma::mat sigma2f_inv_new(n_I, d, arma::fill::none);
  arma::mat sigma2b_inv_new(n_I, d, arma::fill::none);


  // hyperparameter C0
  arma::mat C0f_old(n_I, d, arma::fill::none);
  arma::mat C0f_new(n_I, d, arma::fill::none);

  arma::mat C0b_old(n_I, d, arma::fill::none);
  arma::mat C0b_new(n_I, d, arma::fill::none);

  // PARCOR parameters
  arma::cube betaf_old(n_I, n_I, d, arma::fill::zeros);
  arma::cube betab_old(n_I, n_I, d, arma::fill::zeros);

  arma::cube betaf_new(n_I, n_I, d, arma::fill::zeros);
  arma::cube betab_new(n_I, n_I, d, arma::fill::zeros);

  // local parameters
  arma::cube tau2f_old(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_old(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_new(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_new(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_inv_old(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_inv_old(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_inv_new(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_inv_new(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_log_old(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_log_old(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_log_new(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_log_new(n_I, n_I, d, arma::fill::ones);

  // Global parameters
  arma::vec lambda2f_old(n_I, arma::fill::ones);
  arma::vec lambda2b_old(n_I, arma::fill::ones);

  arma::vec lambda2f_log_old(n_I, arma::fill::ones);
  arma::vec lambda2b_log_old(n_I, arma::fill::ones);

  arma::vec lambda2f_new(n_I, arma::fill::ones);
  arma::vec lambda2b_new(n_I, arma::fill::ones);

  arma::vec lambda2f_log_new(n_I, arma::fill::ones);
  arma::vec lambda2b_log_new(n_I, arma::fill::ones);


  arma::vec a_tauf_new(n_I);
  arma::vec a_taub_new(n_I);

  arma::vec a_tauf_old(n_I);
  arma::vec a_taub_old(n_I);

  // Temp store for parameters
  arma::vec beta_tmp;
  arma::vec beta2_tmp;
  arma::vec sigma2_beta_tmp;

  arma::vec tau2_tmp;
  arma::vec tau2_inv_tmp;
  arma::vec tau2_log_tmp;

  double lambda2_tmp;
  double lambda2_log_tmp;

  double sigma2_tmp;
  double sigma2_inv_tmp;
  double C0_tmp;

  int index;

  // Initial values and objects
  sigma2f_new.fill(1.0);
  sigma2b_new.fill(1.0);
  sigma2f_old.fill(1.0);
  sigma2b_old.fill(1.0);
  sigma2f_inv_old.fill(1.0);
  sigma2b_inv_old.fill(1.0);
  sigma2f_inv_new.fill(1.0);
  sigma2b_inv_new.fill(1.0);

  C0f_new.fill(1.0);
  C0b_new.fill(1.0);
  C0f_old.fill(1.0);
  C0b_old.fill(1.0);

  betaf_new.fill(1.0);
  betab_new.fill(1.0);
  betaf_old.fill(1.0);
  betab_old.fill(1.0);

  tau2f_new.fill(1.0);
  tau2b_new.fill(1.0);
  tau2f_old.fill(1.0);
  tau2b_old.fill(1.0);

  a_tauf_new.fill(a_tau);
  a_taub_new.fill(a_tau);
  a_tauf_old.fill(a_tau);
  a_taub_old.fill(a_tau);


  // definition of non central part cholesky
  arma::mat betaf_chol_old;
  arma::mat betab_chol_old;
  arma::mat betaf_chol_new;
  arma::mat betab_chol_new;


  // definition of tau2 cholesky
  arma::mat tau2f_chol_old;
  arma::mat tau2b_chol_old;

  arma::mat tau2f_chol_new;
  arma::mat tau2b_chol_new;

  arma::mat tau2f_inv_chol_old;
  arma::mat tau2b_inv_chol_old;

  arma::mat tau2f_inv_chol_new;
  arma::mat tau2b_inv_chol_new;

  arma::mat tau2f_log_chol_old;
  arma::mat tau2b_log_chol_old;

  arma::mat tau2f_log_chol_new;
  arma::mat tau2b_log_chol_new;

  // definition temp upper triangular
  arma::mat tmp_upper_triangular;
  arma::mat tmp_beta;
  arma::uvec upper_indices;
  arma::uvec all_indices;
  if(n_I > 1){
    tmp_upper_triangular = arma::mat(n_I, n_I, arma::fill::eye);
    tmp_beta = arma::mat(n_I, n_I);
    upper_indices = arma::trimatu_ind(size(tmp_upper_triangular), 1);
    all_indices = arma::linspace<arma::uvec>(0, n_I*n_I-1, n_I*n_I);
  }


  if(!ind){
    // definition of beta mean
    betaf_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);
    betab_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);

    betaf_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);
    betab_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);

    // definition of tau2 cholesky
    tau2f_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_inv_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_inv_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_inv_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_inv_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_log_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_log_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_log_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_log_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
  }

  // Values to check if the sampler failed or not
  bool succesful = true;
  std::string fail;
  int fail_iter;
  int j = 0;

  // Introduce difference

  bool flag = false;
  // Begin Gibbs loop
  while( !flag && (j < iter_max)){
    for(int m = start; m < d+1; m++){
      for(int k = 0; k < n_I; k++){
        //Rcout << "Step 1" << "\n";
        // Forward
        // ----------------------------
        n_1 = m + 1;
        n_T = N;
        N_m = n_T - n_1 + 1;
        y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        x_tmp = yb.slice(m-1).rows(n_1-m-1, n_T-m-1);
        if(!ind){
          if(k == 1){
            x_tmp = arma::join_rows(x_tmp, -yf.slice(m-1).col(0).rows(n_1-1, n_T-1));
          }else if(k > 1){
            x_tmp = arma::join_rows(x_tmp, -yf.slice(m-1).cols(0, k-1).rows(n_1-1, n_T-1));
          }

        }
        //if(k!=0){
        //  x_tmp = arma::join_rows(x_tmp, yf.slice(m-1).cols(k+1, n_I-1).rows(n_1-1, n_T-1));
        //}
        d_tmp = x_tmp.n_cols;

        beta_tmp = betaf_old.slice(m-1).col(k);
        beta2_tmp = arma::vec(d_tmp, arma::fill::zeros);
        sigma2_beta_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_tmp = arma::vec(d_tmp, arma::fill::zeros);
        tau2_inv_tmp = tau2f_inv_old.slice(m-1).col(k);
        tau2_log_tmp = tau2f_log_old.slice(m-1).col(k);
        sigma2_tmp = sigma2f_old(k, m-1);
        sigma2_inv_tmp = sigma2f_inv_old(k, m-1);
        C0_tmp = C0f_old(k, m-1);
        //Rcout << "Step 2" << "\n";
        if(!ind){
          if( k == 1){
            beta_tmp = arma::vec(d_tmp, arma::fill::zeros);

            tau2_inv_tmp = arma::vec(d_tmp, arma::fill::zeros);
            tau2_log_tmp = arma::vec(d_tmp, arma::fill::zeros);

            beta_tmp(arma::span(0, d_tmp-2)) = betaf_old.slice(m-1).col(k);
            beta_tmp(d_tmp-1) = betaf_chol_old(0, m-1);

            tau2_inv_tmp(arma::span(0, d_tmp-2)) = tau2f_inv_old.slice(m-1).col(k);
            tau2_inv_tmp(d_tmp-1) = tau2f_inv_chol_old(0, m-1);

            tau2_log_tmp(arma::span(0, d_tmp-2)) = tau2f_log_old.slice(m-1).col(k);
            tau2_log_tmp(d_tmp-1) = tau2f_log_chol_old(0, m-1);
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));

            beta_tmp = arma::join_cols(beta_tmp, betaf_chol_old.col(m-1).rows(index, index+k-1));
            tau2_inv_tmp = arma::join_cols(tau2_inv_tmp, tau2f_inv_chol_old.col(m-1).rows(index, index+k-1));
            tau2_log_tmp = arma::join_cols(tau2_log_tmp, tau2f_log_chol_old.col(m-1).rows(index, index+k-1));

          }
        }
        //Rcout << "Step 3" << "\n";
        // update beta mean
        try {
          update_beta(beta_tmp, beta2_tmp, sigma2_beta_tmp,
                      y_tmp, x_tmp, sigma2_inv_tmp, tau2_inv_tmp);
          betaf_new.slice(m-1).col(k) = beta_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              betaf_chol_new(0, m-1) = beta_tmp(n_I);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              betaf_chol_new.col(m-1).rows(index, index+k-1) = beta_tmp(arma::span(n_I, d_tmp-1));
            }
          }

        } catch(...){
          Rcout << "beta problem " << "\n";
          //beta_mean_tmp.fill(nanl(""));
          //beta2_mean_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update forward beta & beta square";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update forward tau2
        try {
          update_local_shrink(tau2_tmp, tau2_inv_tmp, tau2_log_tmp,
                              beta2_tmp, lambda2f_old(k), a_tauf_old(k));
          tau2f_new.slice(m-1).col(k) = tau2_tmp(arma::span(0, n_I-1));
          tau2f_inv_new.slice(m-1).col(k) = tau2_inv_tmp(arma::span(0, n_I-1));
          tau2f_log_new.slice(m-1).col(k) = tau2_log_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              tau2f_chol_new(0, m-1) = arma::as_scalar(tau2_tmp(n_I));
              tau2f_inv_chol_new(0, m-1) = arma::as_scalar(tau2_inv_tmp(n_I));
              tau2f_log_chol_new(0, m-1) = arma::as_scalar(tau2_log_tmp(n_I));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              tau2f_chol_new.col(m-1).rows(index, index+k-1) = tau2_tmp(arma::span(n_I, d_tmp-1));
              tau2f_inv_chol_new.col(m-1).rows(index, index+k-1) = tau2_inv_tmp(arma::span(n_I, d_tmp-1));
              tau2f_log_chol_new.col(m-1).rows(index, index+k-1) = tau2_log_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...) {
          //tau2_tmp.fill(nanl(""));
          //tau2_inv_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update forward tau2, tau2_inv & tau2_log";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update forward variance sigma2
        //Rcout << "Step 5" << "\n";
        try{
          update_sigma2(beta_tmp, sigma2_beta_tmp, y_tmp, x_tmp, sigma2_tmp, sigma2_inv_tmp, c0, C0_tmp);
          sigma2f_new(k, m-1) = sigma2_tmp;
          sigma2f_inv_new(k, m-1) = sigma2_inv_tmp;
        } catch(...){
          if(succesful == true){
            fail = "update forward sigma2 & sigma2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }


        // update forward C0
        //Rcout << "Step 6" << "\n";
        try{
          update_C0(C0_tmp, sigma2_inv_tmp, c0, g0, G0);
          C0f_new(k, m-1) = C0_tmp;
        } catch(...){
          if(succesful == true){
            fail = "update forward C0";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update forward prediction error
        yf.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp - x_tmp * beta_tmp;

        // Backward
        // --------------------------------
        //Rcout << "Step 7" << "\n";
        n_1 = 1;          // backward index
        n_T = N - m;      // backward index
        N_m = n_T - n_1 + 1;

        y_tmp = yb.slice(m-1).col(k).rows(n_1-1, n_T-1);
        x_tmp = yf.slice(m-1).rows(n_1+m-1, n_T+m-1);

        if(!ind){
          if(k == 1){
            x_tmp = arma::join_rows(x_tmp, -yb.slice(m-1).col(0).rows(n_1-1, n_T-1));
          }else if(k > 1){
            x_tmp = arma::join_rows(x_tmp, -yb.slice(m-1).cols(0, k-1).rows(n_1-1, n_T-1));
          }
        }

        d_tmp = x_tmp.n_cols;

        beta_tmp = betab_old.slice(m-1).col(k);
        beta2_tmp = arma::vec(d_tmp, arma::fill::zeros);
        sigma2_beta_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_tmp = arma::vec(d_tmp, arma::fill::zeros);
        tau2_inv_tmp = tau2b_inv_old.slice(m-1).col(k);
        tau2_log_tmp = tau2b_log_old.slice(m-1).col(k);
        sigma2_tmp = sigma2b_old(k, m-1);
        sigma2_inv_tmp = sigma2b_inv_old(k, m-1);
        C0_tmp = C0b_old(k, m-1);
        //Rcout << "Step 8" << "\n";
        if(!ind){
          if( k == 1){
            beta_tmp = arma::vec(d_tmp, arma::fill::zeros);
            tau2_inv_tmp = arma::vec(d_tmp, arma::fill::zeros);
            tau2_log_tmp = arma::vec(d_tmp, arma::fill::zeros);

            beta_tmp(arma::span(0, d_tmp-2)) = betab_old.slice(m-1).col(k);
            beta_tmp(d_tmp-1) = betab_chol_old(0, m-1);

            tau2_inv_tmp(arma::span(0, d_tmp-2)) = tau2b_inv_old.slice(m-1).col(k);
            tau2_inv_tmp(d_tmp-1) = tau2b_inv_chol_old(0, m-1);

            tau2_log_tmp(arma::span(0, d_tmp-2)) = tau2b_log_old.slice(m-1).col(k);
            tau2_log_tmp(d_tmp-1) = tau2b_log_chol_old(0, m-1);

          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            beta_tmp = arma::join_cols(beta_tmp, betab_chol_old.col(m-1).rows(index, index+k-1));
            tau2_inv_tmp = arma::join_cols(tau2_inv_tmp, tau2b_inv_chol_old.col(m-1).rows(index, index+k-1));
            tau2_log_tmp = arma::join_cols(tau2_log_tmp, tau2b_log_chol_old.col(m-1).rows(index, index+k-1));
          }
        }


        //Rcout << "Step 9" << "\n";
        // update beta mean
        try {
          update_beta(beta_tmp, beta2_tmp, sigma2_beta_tmp,
                      y_tmp, x_tmp, sigma2_inv_tmp, tau2_inv_tmp);
          betab_new.slice(m-1).col(k) = beta_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              betab_chol_new(0, m-1) = beta_tmp(n_I);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              betab_chol_new.col(m-1).rows(index, index+k-1) = beta_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...){
          //beta_mean_tmp.fill(nanl(""));
          //beta2_mean_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward beta & beta square";
            fail_iter = j + 1;
            succesful = false;
          }
        }


        //
        // update backward tau2
        //Rcout << "Step 10" << "\n";
        try {
          update_local_shrink(tau2_tmp, tau2_inv_tmp, tau2_log_tmp, beta2_tmp, lambda2b_old(k), a_taub_old(k));
          tau2b_new.slice(m-1).col(k) = tau2_tmp(arma::span(0, n_I-1));
          tau2b_inv_new.slice(m-1).col(k) = tau2_inv_tmp(arma::span(0, n_I-1));
          tau2b_log_new.slice(m-1).col(k) = tau2_log_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              tau2b_chol_new(0, m-1) = arma::as_scalar(tau2_tmp(n_I));
              tau2b_inv_chol_new(0, m-1) = arma::as_scalar(tau2_inv_tmp(n_I));
              tau2b_log_chol_new(0, m-1) = arma::as_scalar(tau2_log_tmp(n_I));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              tau2b_chol_new.col(m-1).rows(index, index+k-1) = tau2_tmp(arma::span(n_I, d_tmp-1));
              tau2b_inv_chol_new.col(m-1).rows(index, index+k-1) = tau2_inv_tmp(arma::span(n_I, d_tmp-1));
              tau2b_log_chol_new.col(m-1).rows(index, index+k-1) = tau2_log_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...) {
          //tau2_tmp.fill(nanl(""));
          //tau2_inv_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward tau2, tau2_inv & tau2_log";
            fail_iter = j + 1;
            succesful = false;
          }
        }


        //Rcout << "Step 11" << "\n";
        // update backward variance sigma2
        try{
          update_sigma2(beta_tmp, sigma2_beta_tmp, y_tmp, x_tmp, sigma2_tmp, sigma2_inv_tmp, c0, C0_tmp);
          sigma2b_new(k, m-1) = sigma2_tmp;
          sigma2b_inv_new(k, m-1) = sigma2_inv_tmp;
        } catch(...){
          if(succesful == true){
            fail = "update backward sigma2 & sigma2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        //Rcout << "Step 12" << "\n";
        // update forward C0
        try{
          update_C0(C0_tmp, sigma2_inv_tmp, c0, g0, G0);
          C0b_new(k, m-1) = C0_tmp;
        } catch(...){
          if(succesful == true){
            fail = "update backward C0";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        //Rcout << "Step 12" << "\n";
        // update backward prediction error
        yb.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp - x_tmp * beta_tmp;
      }
      //Rcout << "Step 13" << "\n";
      // transform back
      if(!ind){
        // forward part
        n_1 = m + 1;
        n_T = N;
        tmp_upper_triangular = arma::mat(n_I, n_I, arma::fill::eye);
        tmp_upper_triangular.elem(upper_indices) = arma::trans(betaf_chol_new.col(m-1));

        for(int i = n_1-1; i < n_T; i++){
          yf.slice(m).row(i) = arma::trans(arma::inv(tmp_upper_triangular.t())*arma::trans(yf.slice(m).row(i)));
        }
        //backward part
        n_1 = 1;          // backward index
        n_T = N - m;      // backward index
        tmp_upper_triangular = arma::mat(n_I, n_I, arma::fill::eye);
        tmp_upper_triangular.elem(upper_indices) = arma::trans(betab_chol_new.col(m-1));
        for(int i = n_1-1; i < n_T; i++){
          yb.slice(m).row(i) = arma::trans(arma::inv(tmp_upper_triangular.t())*arma::trans(yb.slice(m).row(i)));
        }
      }
    }
    //Rcout << "Step 14" << "\n";
    //Rcout << "forward lambda2" << "\n";
    for(int k = 0; k < n_I; k++){
      try {
        tau2_tmp = arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
        if(!ind){
          if(k == 1){
            tau2_tmp = arma::join_cols(arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_chol_new(0, arma::span(start-1, d-1))));
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            tau2_tmp = arma::join_cols(arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
          }
        }
        update_global_shrink(tau2_tmp,lambda2_tmp, lambda2_log_tmp, a_tauf_old(k), e1, e2);
        lambda2f_new(k) = lambda2_tmp;
        lambda2f_log_new(k) = lambda2_log_tmp;
      } catch (...) {
        //lambda2f_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update forward lambda2 & lambda2_log";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }
    //Rcout << "Step 15" << "\n";
    //Rcout << "backward lambda2" << "\n";
    for(int k = 0; k < n_I; k++){
      try {
        tau2_tmp = arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
        if(!ind){
          if(k == 1){
            tau2_tmp = arma::join_cols(arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_chol_new(0, arma::span(start-1, d-1))));
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            tau2_tmp = arma::join_cols(arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
          }
        }
        update_global_shrink(tau2_tmp, lambda2_tmp, lambda2_log_tmp, a_taub_old(k), e1, e2);
        lambda2b_new(k) = lambda2_tmp;
        lambda2b_log_new(k) = lambda2_log_tmp;
      } catch (...) {
        //lambda2b_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update backward lambda2 & lambda2_log";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }
    //Rcout << "Step 16" << "\n";
    if(learn_a_tau){
      // Rcout << "forward a_tau" << "\n";
      for(int k = 0; k < n_I; k++){
        try{
          tau2_tmp = arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
          tau2_log_tmp = arma::vectorise(tau2f_log_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
          if(!ind){
            if(k == 1){
              tau2_tmp = arma::join_cols(arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_chol_new(0, arma::span(start-1, d-1))));
              tau2_log_tmp = arma::join_cols(arma::vectorise(tau2f_log_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_log_chol_new(0, arma::span(start-1, d-1))));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              tau2_tmp = arma::join_cols(arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
              tau2_log_tmp = arma::join_cols(arma::vectorise(tau2f_log_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_log_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
            }
          }
          a_tauf_new(k) = DG_approx(tau2_tmp, tau2_log_tmp, lambda2f_new(k), lambda2f_log_new(k), b_tau, sample_size);
        }catch (...){
          if (succesful == true){
            fail = "update forward a_tau";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }
      // Rcout << "backward a_tau" << "\n";
      for(int k = 0; k < n_I; k++){
        try{
          tau2_tmp = arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
          tau2_log_tmp = arma::vectorise(tau2b_log_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
          if(!ind){
            if(k == 1){
              tau2_tmp = arma::join_cols(arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_chol_new(0, arma::span(start-1, d-1))));
              tau2_log_tmp = arma::join_cols(arma::vectorise(tau2b_log_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_log_chol_new(0, arma::span(start-1, d-1))));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              tau2_tmp = arma::join_cols(arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
              tau2_log_tmp = arma::join_cols(arma::vectorise(tau2b_log_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_log_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
            }
          }
          a_taub_new(k) = DG_approx(tau2_tmp, tau2_log_tmp, lambda2b_new(k), lambda2b_log_new(k), b_tau, sample_size);
        }catch (...){
          if (succesful == true){
            fail = "update backward a_tau";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }
    }
    //Rcout << "Step 17" << "\n";
    // Updating stop criterion
    // update old state of sigma2
    sigma2f_old = sigma2f_new;
    sigma2b_old = sigma2b_new;

    // update old state of C0
    C0f_old = C0f_new;
    C0b_old = C0b_new;

    // update tau2
    tau2f_old = tau2f_new;
    tau2b_old = tau2b_new;
    tau2f_inv_old = tau2f_inv_new;
    tau2b_inv_old = tau2b_inv_new;
    tau2f_log_old = tau2f_log_new;
    tau2b_log_old = tau2b_log_new;

    // update lambda2
    lambda2f_old = lambda2f_new;
    lambda2b_old = lambda2b_new;
    lambda2f_log_old = lambda2f_log_new;
    lambda2b_log_old = lambda2b_log_new;

    // update a tau
    a_tauf_old = a_tauf_new;
    a_taub_old = a_taub_new;

    if(!ind){

      tau2f_chol_old = tau2f_chol_new;
      tau2b_chol_old = tau2b_chol_new;
      tau2f_inv_chol_old = tau2f_inv_chol_new;
      tau2b_inv_chol_old = tau2b_inv_chol_new;
      tau2f_log_chol_old = tau2f_log_chol_new;
      tau2b_log_chol_old = tau2b_log_chol_new;
    }

    flag = true;
    counts = 0;
    for(int m = start-1; m < d; m++){
      if(!ind){
        // forward part
        tmp_upper_triangular = arma::mat(n_I, n_I, arma::fill::eye);
        tmp_upper_triangular.elem(upper_indices) = betaf_chol_new.col(m);
        betaf_new.slice(m) = betaf_new.slice(m)*arma::inv(tmp_upper_triangular);
        // backward part
        tmp_upper_triangular = arma::mat(n_I, n_I, arma::fill::eye);
        tmp_upper_triangular.elem(upper_indices) = betab_chol_new.col(m);
        betab_new.slice(m) = betab_new.slice(m)*arma::inv(tmp_upper_triangular);
      }
      //diff(counts) = compute_norm_matrix(betaf_new.slice(m).rows(d, N-d-1), betaf_old.slice(m).rows(d, N-d-1));
      diff(counts) = arma::norm(betaf_new.slice(m)-betaf_old.slice(m), 2);
      flag = (diff(counts) < epsilon) && flag;
      counts += 1;

      //diff(counts) = compute_norm_matrix(betab_new.slice(m).rows(d, N-d-1), betab_old.slice(m).rows(d, N-d-1));
      diff(counts) = arma::norm(betab_new.slice(m)-betab_old.slice(m), 2);
      flag = (diff(counts) < epsilon) && flag;
      counts += 1;
    }




    betaf_old = betaf_new;
    betab_old = betab_new;
    if(!ind){
      betaf_chol_old = betaf_chol_new;
      betab_chol_old = betab_chol_new;
    }




    // Increment progress bar
    if (arma::any(prog_rep_points == j)) {
      p.increment();
    }
    //Rcout << "iteration:" << j << "\n";

    j += 1;
  }





  // return everything as a nested list (due to size restrictions on Rcpp::Lists)
  if(ind){
    return Rcpp::List::create(_["beta"] = List::create(_["f"] = betaf_old, _["b"] = betab_old),
                              _["tau2"] = List::create(_["f"] = tau2f_old, _["b"] = tau2b_old),
                              _["lambda2"] = List::create(_["f"] = lambda2f_old, _["b"] = lambda2b_old),
                              _["a_tau"] = List::create(_["f"] = a_tauf_old, _["b"] = a_taub_old),
                              _["SIGMA"] = List::create(_["f"] = sigma2f_old, _["b"] = sigma2b_old),
                              _["C0"] = List::create(_["f"] = C0f_old, _["b"] = C0b_old),
                              _["iter"] = j,
                              _["diff"] = diff,
                              _["success_vals"] = List::create(
                                _["success"] = succesful,
                                _["fail"] = fail,
                                _["fail_iter"] = fail_iter)
    );
  }else{
    return Rcpp::List::create(_["beta"] = List::create(_["f"] = betaf_old, _["b"] = betab_old),
                              _["beta_chol"] = List::create(_["f"] = betaf_chol_old, _["b"] = betab_chol_old),
                              _["tau2"] = List::create(_["f"] = tau2f_old, _["b"] = tau2b_old),
                              _["lambda2"] = List::create(_["f"] = lambda2f_old, _["b"] = lambda2b_old),
                              _["a_tau"] = List::create(_["f"] = a_tauf_old, _["b"] = a_taub_old),
                              _["SIGMA"] = List::create(_["f"] = sigma2f_old, _["b"] = sigma2b_old),
                              _["C0"] = List::create(_["f"] = C0f_old, _["b"] = C0b_old),
                              _["iter"] = j,
                              _["diff"] = diff,
                              _["success_vals"] = List::create(
                                _["success"] = succesful,
                                _["fail"] = fail,
                                _["fail_iter"] = fail_iter)
    );
  }
}

