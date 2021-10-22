#####################################
####### run shrinkage PARCOR ########
#####################################

mcmc_sPARCOR <- function(y,
                         d,
                         niter = 10000,
                         nburn = round(niter / 2),
                         nthin = 1,
                         learn_a_xi = TRUE,
                         learn_a_tau = TRUE,
                         a_xi = 0.1,
                         a_tau = 0.1,
                         learn_kappa2 = TRUE,
                         learn_lambda2 = TRUE,
                         kappa2 = 20,
                         lambda2 = 20,
                         hyperprior_param,
                         display_progress = TRUE,
                         ret_beta_nc = FALSE,
                         S_0,
                         delta,
                         uncertainty=FALSE,
                         ind = TRUE,
                         skip = TRUE,
                         cpus = 1,
                         sv,
                         sv_param,
                         MH_tuning){

  # number of time series
  K <- dim(y)[2]
  # number of time points
  n_t <- dim(y)[1]


  # Input checking ----------------------------------------------------------
  # default hyperparameter values
  default_hyper <- list(c0 = 2.5,
                        g0 = 5,
                        G0 = 5 / (2.5 - 1),
                        e1 = 0.001,
                        e2 = 0.001,
                        d1 = 0.001,
                        d2 = 0.001,
                        beta_a_xi = 10,
                        beta_a_tau = 10,
                        alpha_a_xi = 5,
                        alpha_a_tau = 5)

  # default sv params
  default_hyper_sv <- list(Bsigma_sv = 1,
                           a0_sv = 5,
                           b0_sv = 1.5,
                           bmu = 0,
                           Bmu = 1)

  # default tuning parameters
  default_tuning_par <- list(a_xi_adaptive = TRUE,
                             a_xi_tuning_par = 1,
                             a_xi_target_rate = 0.44,
                             a_xi_max_adapt = 0.01,
                             a_xi_batch_size = 50,
                             a_tau_adaptive = TRUE,
                             a_tau_tuning_par = 1,
                             a_tau_target_rate = 0.44,
                             a_tau_max_adapt = 0.01,
                             a_tau_batch_size = 50)
  # Change hyperprior values if user overwrites them
  if (missing(hyperprior_param)){
    hyperprior_param <- default_hyper
  }

  # Change sv parameter values if user overwrites them
  if(missing(sv_param) | sv == FALSE){
    sv_param <- default_hyper_sv
  }

  # Change tuning parameter values if user overwrites them
  if (missing(MH_tuning)){
    MH_tuning <- default_tuning_par
  }
  # # Check if all numeric inputs are correct
  # to_test_num <- list(lambda2 = lambda2,
  #                     kappa2 = kappa2,
  #                     a_xi = a_xi,
  #                     a_tau = a_tau,
  #                     a_tuning_par_xi = MH_tuning$a_tuning_par_xi,
  #                     a_tuning_par_tau = MH_tuning$a_tuning_par_tau)
  #
  # if (missing(hyperprior_param) == FALSE){
  #   to_test_num <- c(to_test_num, hyperprior_param)
  # }



  if ((niter - nburn) < 2){
    stop("niter has to be larger than or equal to nburn + 2")
  }

  if (nthin == 0){
    stop("nthin can not be 0")
  }

  if ((niter - nburn)/2 < nthin){
    stop("nthin can not be larger than (niter - nburn)/2")
  }

  if(skip){
    ## skip the first stage,

    phi_fwd <- array(0, dim = c(n_t, K^2))
    phi_bwd <- array(0, dim = c(n_t, K^2))

    if(missing(delta)){
      ### Set up discount ###
      grid_seq <- seq(0.95, 1, 0.01)
      tmp <- as.matrix(expand.grid(grid_seq, grid_seq))
      tmp_dim <- dim(tmp)
      delta <- array(dim = c(tmp_dim[1], K^2, 1))
    }

    ## run conjugate prior structure at the first stage
    result_skip <- run_parcor_parallel(F1 = t(y),
                                       delta = delta,
                                       P = 1,
                                       S_0 = S_0*diag(K),
                                       sample_size = (niter-nburn)/nthin,
                                       DIC = FALSE,
                                       uncertainty = TRUE)
    ## run sampler
    runtime <- system.time({
      suppressWarnings({
        result <- do_mcmc_sPARCOR(t(result_skip$F1_fwd),
                                  t(result_skip$F1_bwd),
                                  d,
                                  niter,
                                  nburn,
                                  nthin,
                                  hyperprior_param$c0,
                                  hyperprior_param$g0,
                                  hyperprior_param$G0,
                                  hyperprior_param$d1,
                                  hyperprior_param$d2,
                                  hyperprior_param$e1,
                                  hyperprior_param$e2,
                                  learn_lambda2,
                                  learn_kappa2,
                                  lambda2,
                                  kappa2,
                                  learn_a_xi,
                                  learn_a_tau,
                                  a_xi,
                                  a_tau,
                                  MH_tuning$a_xi_tuning_par,
                                  MH_tuning$a_tau_tuning_par,
                                  hyperprior_param$beta_a_xi,
                                  hyperprior_param$beta_a_tau,
                                  hyperprior_param$alpha_a_xi,
                                  hyperprior_param$alpha_a_tau,
                                  display_progress,
                                  ret_beta_nc,
                                  ind,
                                  skip,
                                  sv,
                                  sv_param$Bsigma_sv,
                                  sv_param$a0_sv,
                                  sv_param$b0_sv,
                                  sv_param$bmu,
                                  sv_param$Bmu,
                                  unlist(MH_tuning[grep("adaptive", names(MH_tuning))]),
                                  unlist(MH_tuning[grep("target", names(MH_tuning))]),
                                  unlist(MH_tuning[grep("max", names(MH_tuning))]),
                                  unlist(MH_tuning[grep("size", names(MH_tuning))]))
      })
    })

  # Throw an error if the sampler failed
    if (result$success_vals$success == FALSE){
      stop(paste0("The sampler failed at iteration ",
                  result$success_vals$fail_iter,
                  " while trying to ",
                  result$success_vals$fail, ". ",
                  "Try rerunning the model. ",
                  "If the sampler fails again, try changing the prior to be more informative. ",
                  "If the problem still persists, please contact the maintainer: ",
                  maintainer("shrinkTVP")))
    } else {
      result$success_vals <- NULL
    }

    # Post process sampler results --------------------------------------------
    for(i in 1:((niter - nburn)/nthin)){
      for(j in (1+1):(n_t-1)){
        phi_fwd[j, ] <- tryCatch(rmvn(n = 1, mu = as.vector(result_skip$phi_fwd[, j, 1]),
                             sigma = result_skip$Cnt_fwd[[1]][[j]]), error = function(e){as.vector(result_skip$phi_fwd[,j,1])})
        phi_bwd[j, ] <- tryCatch(rmvn(n = 1, mu = as.vector(result_skip$phi_bwd[, j, 1]),
                             sigma = result_skip$Cnt_bwd[[1]][[j]]), error = function(e){as.vector(result_skip$phi_bwd[,j,1])})
      }
      result$beta$f[[i]][, , 1] <- phi_fwd
      result$beta$b[[i]][, , 1] <- phi_bwd

      result$beta$f[[i]] <- aperm(result$beta$f[[i]], perm = c(2,1,3))
      result$beta$b[[i]] <- aperm(result$beta$b[[i]], perm = c(2,1,3))
    }
  }else{
    runtime <- system.time({
      suppressWarnings({
        result <- do_mcmc_sPARCOR(y,
                                  y,
                                  d,
                                  niter,
                                  nburn,
                                  nthin,
                                  hyperprior_param$c0,
                                  hyperprior_param$g0,
                                  hyperprior_param$G0,
                                  hyperprior_param$d1,
                                  hyperprior_param$d2,
                                  hyperprior_param$e1,
                                  hyperprior_param$e2,
                                  learn_lambda2,
                                  learn_kappa2,
                                  lambda2,
                                  kappa2,
                                  learn_a_xi,
                                  learn_a_tau,
                                  a_xi,
                                  a_tau,
                                  MH_tuning$a_xi_tuning_par,
                                  MH_tuning$a_tau_tuning_par,
                                  hyperprior_param$beta_a_xi,
                                  hyperprior_param$beta_a_tau,
                                  hyperprior_param$alpha_a_xi,
                                  hyperprior_param$alpha_a_tau,
                                  display_progress,
                                  ret_beta_nc,
                                  ind,
                                  skip,
                                  sv,
                                  sv_param$Bsigma_sv,
                                  sv_param$a0_sv,
                                  sv_param$b0_sv,
                                  sv_param$bmu,
                                  sv_param$Bmu,
                                  unlist(MH_tuning[grep("adaptive", names(MH_tuning))]),
                                  unlist(MH_tuning[grep("target", names(MH_tuning))]),
                                  unlist(MH_tuning[grep("max", names(MH_tuning))]),
                                  unlist(MH_tuning[grep("size", names(MH_tuning))]))
      })
    })

    for(i in 1:((niter - nburn)/nthin)){
      result$beta$f[[i]] <- aperm(result$beta$f[[i]], perm = c(2,1,3))
      result$beta$b[[i]] <- aperm(result$beta$b[[i]], perm = c(2,1,3))
    }
  }


  if (display_progress == TRUE){
    cat("Timing (elapsed): ", file = stderr())
    cat(runtime["elapsed"], file = stderr())
    cat(" seconds.\n", file = stderr())
    cat(round( (niter + nburn) / runtime[3]), "iterations per second.\n\n", file = stderr())
    cat("Converting results to coda objects and summarizing draws... ", file = stderr())
  }

  ### transform PARCOR coefficients to TVVAR coefficients
  phi_fwd <- result$beta$f
  phi_bwd <- result$beta$b
  sfInit(parallel = TRUE, cpus = cpus, type = "SOCK")
  sfLibrary(PARCOR)
  sfExport("phi_fwd", "phi_bwd", "K", "d")
  ar_coef_sample <- sfLapply(1:((niter-nburn)/nthin), function(i) obtain_TVAR(result$beta$f[[i]], result$beta$b[[i]], K, d))
  sfStop()
  if(uncertainty){
    return(list(phi_fwd = result$beta$f,
                phi_bwd = result$beta$b,
                chol_fwd = result$beta_chol$f,
                chol_bwd = result$beta_chol$b,
                ar = ar_coef_sample,
                SIGMA = result$SIGMA$f,
                runtime = runtime,
                MH_diag = result$MH_diag,
                a_xi = result$a_xi,
                a_tau = result$a_tau))
  }else{
    ### extract forward part
    phi_fwd <- apply(simplify2array(result$beta$f), 1:3, mean)
    phi_fwd <- aperm(phi_fwd, perm = c(2, 1, 3))
    if(!ind){
      beta_chol_fwd <- apply(simplify2array(result$beta_chol$f), 1:3, mean)
    }

    ### extract backward part
    phi_bwd <- apply(simplify2array(result$beta$b), 1:3, mean)
    phi_bwd <- aperm(phi_bwd, perm = c(2, 1, 3))
    if(!ind){
      beta_chol_bwd <- apply(simplify2array(result$beta_chol$b), 1:3, mean)
    }

    ### extract forward SIGMA
    SIGMA <- apply(simplify2array(result$SIGMA$f), 1:3, mean)

    ### transfer PARCOR coefficients to AR coefficients
    ar <- apply(simplify2array(ar_coef_sample), 1:3, mean)
    return(list(phi_fwd = phi_fwd,
                phi_bwd = phi_bwd,
                phi_chol_fwd = beta_chol_fwd,
                phi_chol_bwd = beta_chol_bwd,
                SIGMA = SIGMA,
                ar = ar,
                runtime = runtime,
                MH_diag = result$MH_diag,
                a_xi = result$a_xi,
                a_tau = result$a_tau))
  }
}
