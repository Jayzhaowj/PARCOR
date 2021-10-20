vi_sPARCOR <- function(y, d, a_xi, a_tau, learn_a_xi, learn_a_tau, hyperprior_param,
                       iter_max, ind, S_0, epsilon, skip, delta, sample_size = 1000){
  K <- dim(y)[2]
  n_t <- dim(y)[1]

  # Input checking ------------------------------------------------
  # default hyperparameter values
  default_hyper <- list(e1 = 0.001,
                        e2 = 0.001,
                        d1 = 0.001,
                        d2 = 0.001,
                        b_xi = 10,
                        b_tau = 10)
  if (missing(hyperprior_param)){
    hyperprior_param <- default_hyper
  }

  if(skip){
    ## skip the first stage by using PARCOR model
    if(missing(delta)){
      ### Set up discount ###
      grid_seq <- seq(0.95, 1, 0.01)
      tmp <- as.matrix(grid_seq)
      tmp_dim <- dim(tmp)
      delta <- array(dim = c(tmp_dim[1], K^2, 1))
    }
    result_skip <- run_parcor(F1 = t(y),
                              delta = delta,
                              P = 1,
                              S_0 = S_0*diag(K),
                              DIC = FALSE,
                              uncertainty = FALSE)

    result <- vi_shrinkTVP(y_fwd = t(result_skip$F1_fwd),
                           y_bwd = t(result_skip$F1_bwd),
                           d = d,
                           d1 = hyperprior_param$d1,
                           d2 = hyperprior_param$d2,
                           e1 = hyperprior_param$e1,
                           e2 = hyperprior_param$e2,
                           a_xi = a_xi,
                           a_tau = a_tau,
                           learn_a_xi = learn_a_xi,
                           learn_a_tau = learn_a_tau,
                           iter_max = iter_max,
                           ind = ind,
                           S_0 = S_0,
                           epsilon = epsilon,
                           skip = skip,
                           sample_size = sample_size,
                           b_xi = hyperprior_param$b_xi,
                           b_tau = hyperprior_param$b_tau)

    result$beta$f[, , 1] <- t(result_skip$phi_fwd[, , 1])
    result$beta$b[, , 1] <- t(result_skip$phi_bwd[, , 1])
  }else{
    result <- vi_shrinkTVP(y_fwd = y,
                           y_bwd = y,
                           d = d,
                           d1 = hyperprior_param$d1,
                           d2 = hyperprior_param$d2,
                           e1 = hyperprior_param$e1,
                           e2 = hyperprior_param$e2,
                           a_xi = a_xi,
                           a_tau = a_tau,
                           learn_a_xi = learn_a_xi,
                           learn_a_tau = learn_a_tau,
                           iter_max = iter_max,
                           ind = ind,
                           S_0 = S_0,
                           epsilon = epsilon,
                           skip = skip,
                           sample_size = sample_size,
                           b_xi = hyperprior_param$b_xi,
                           b_tau = hyperprior_param$b_tau)
  }
  return(result)
}

