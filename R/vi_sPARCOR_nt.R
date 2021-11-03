vi_sPARCOR_NT <- function(y, d, a_tau, learn_a_tau, hyperprior_param,
                          iter_max, ind, epsilon, sample_size = 1000){
  K <- dim(y)[2]
  n_t <- dim(y)[1]

  # Input checking ------------------------------------------------
  # default hyperparameter values
  default_hyper <- list(e1 = 0.001,
                        e2 = 0.001,
                        b_tau = 10,
                        c0 = 2.5,
                        g0 = 5)
  default_hyper$G0 <- default_hyper$g0*(default_hyper$c0 - 1)
  if (missing(hyperprior_param)){
    hyperprior_param <- default_hyper
  }

  result <- vi_shrinkNTVP(y_fwd = y,
                         y_bwd = y,
                         d = d,
                         e1 = hyperprior_param$e1,
                         e2 = hyperprior_param$e2,
                         c0 = hyperprior_param$c0,
                         g0 = hyperprior_param$g0,
                         G0 = hyperprior_param$G0,
                         a_tau = a_tau,
                         learn_a_tau = learn_a_tau,
                         iter_max = iter_max,
                         ind = ind,
                         epsilon = epsilon,
                         sample_size = sample_size,
                         b_tau = hyperprior_param$b_tau)


  return(result)
}

