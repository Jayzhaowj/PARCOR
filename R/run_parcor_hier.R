#############################################
###### run dynamic hierarchical PARCOR model
#############################################

hparcor <- function(yt, delta, P,
                    sample_size = 1000L,
                    chains = 1, DIC = TRUE, uncertainty = TRUE){
  ### number of time point
  n_t <- dim(yt)[1]
  ### number of time series
  n_I <- dim(yt)[2]
  ### number of discount factor options
  num_delta <- dim(delta)[1]

  ### generate matrix F2
  F2_m <- data.frame(a=gl(n_I, 1))
  F2 <- as.matrix(model.matrix(~a, F2_m, contrasts=list(a="contr.sum")))


  ### default delta
  seq_delta <- seq(0.99, 0.999, by = 0.001)
  default_delta <- as.matrix(expand.grid(seq_delta, seq_delta))
  if(missing(delta)){
    delta <- default_delta
  }

  ### storage of variables
  resid_fwd <- array(0, dim = c(n_t, n_I, P+1))
  resid_bwd <- array(0, dim = c(n_t, n_I, P+1))

  phi_fwd <- array(0, dim = c(n_I, n_t, P))
  phi_bwd <- array(0, dim = c(n_I, n_t, P))

  mu_fwd <- array(0, dim = c(n_I, n_t, P))
  mu_bwd <- array(0, dim = c(n_I, n_t, P))

  sigma2t_fwd <- matrix(0, nrow = n_t, ncol = P)
  sigma2t_bwd <- matrix(0, nrow = n_t, ncol = P)

  best_delta_fwd <- matrix(0, nrow = P, ncol = 2)
  best_delta_bwd <- matrix(0, nrow = P, ncol = 2)

  best_pred_dens_fwd <- numeric(P)
  best_pred_dens_bwd <- numeric(P)

  pDIC_fwd <- numeric(P)
  pDIC_bwd <- numeric(P)

  phi_fwd_sample <- array(NA, dim = c(sample_size, n_I, n_t, P))
  phi_bwd_sample <- array(NA, dim = c(sample_size, n_I, n_t, P))

  mu_fwd_sample <- array(NA, dim = c(sample_size, n_I, n_t, P))
  mu_bwd_sample <- array(NA, dim = c(sample_size, n_I, n_t, P))

  ### initialization
  resid_fwd[, , 1] <- yt
  resid_bwd[, , 1] <- yt

  for(j in 1:P){
    ## forward
    best_fwd <- ffbs_DIC(yt = t(resid_fwd[, , j]),
                         F1 = t(resid_bwd[, , j]), F2 = F2,
                         n_t = n_t, n_I = n_I, m = j, type = 1, P = P,
                         delta = delta, DIC = DIC, sample_size = sample_size,
                         chains = chains, uncertainty=uncertainty)

    ## backward
    best_bwd <- ffbs_DIC(yt = t(resid_bwd[, , j]),
                         F1 = t(resid_fwd[, , j]), F2 = F2,
                         n_t = n_t, n_I = n_I, m = j, type = 0, P = P,
                         delta = delta, DIC = DIC, sample_size = sample_size,
                         chains = chains, uncertainty=uncertainty)



    if(uncertainty){
      phi_fwd_sample[, , , j] <- best_fwd$mnt_sample
      phi_bwd_sample[, , , j] <- best_bwd$mnt_sample
      mu_fwd_sample[, , , j] <- best_fwd$mnkt_sample
      mu_bwd_sample[, , , j] <- best_bwd$mnkt_sample
    }

    ## retrieve the results
    pDIC_fwd[j] <- best_fwd$pDIC
    pDIC_bwd[j] <- best_bwd$pDIC
    best_delta_fwd[j, ] <- best_fwd$delta
    best_delta_bwd[j, ] <- best_bwd$delta
    best_pred_dens_fwd[j] <- best_fwd$ll
    best_pred_dens_bwd[j] <- best_bwd$ll
    phi_fwd[, , j] <- best_fwd$mnt
    phi_bwd[, , j] <- best_bwd$mnt
    mu_fwd[, , j] <- best_fwd$mnkt
    mu_bwd[, , j] <- best_bwd$mnkt
    sigma2t_fwd[, j] <- best_fwd$sigma2t
    sigma2t_bwd[, j] <- best_bwd$sigma2t
    DIC_fwd <- 2*(cumsum(pDIC_fwd) - best_pred_dens_fwd)
    DIC_bwd <- 2*(cumsum(pDIC_bwd) - best_pred_dens_bwd)

    ### obtain (j+1) residuals
    resid_fwd[, , j+1] <- t(best_fwd$residuals)
    resid_bwd[, , j+1] <- t(best_bwd$residuals)

    cat('\nThe current iteration: ', j, "/", P)


  }
  return(list(resid_fwd = resid_fwd, resid_bwd = resid_bwd,
              phi_fwd = phi_fwd, phi_bwd = phi_bwd,
              mu_fwd = mu_fwd, mu_bwd = mu_bwd,
              sigma2t_fwd = sigma2t_fwd, sigma2t_bwd = sigma2t_bwd,
              best_delta_fwd = best_delta_fwd,
              best_delta_bwd = best_delta_bwd,
              best_pred_dens_fwd = best_pred_dens_fwd,
              best_pred_dens_bwd = best_pred_dens_bwd,
              pDIC_fwd = pDIC_fwd,
              pDIC_bwd = pDIC_bwd,
              DIC_fwd = DIC_fwd,
              DIC_bwd = DIC_bwd,
              phi_fwd_sample = phi_fwd_sample,
              phi_bwd_sample = phi_bwd_sample,
              mu_fwd_sample = mu_fwd_sample,
              mu_bwd_sample = mu_bwd_sample))
}


compute_TVAR_hier <- function(phi_fwd, phi_bwd, P_opt){
  result <- run_dl(phi_fwd = phi_fwd,
                          phi_bwd = phi_bwd)
  return(result[[P_opt]]$forward)
}

