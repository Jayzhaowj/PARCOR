##### compute quantile of spectral density #######
sample_sd <- function(phi, SIGMA, P_max,
                      start = 0.001, end = 0.499,
                      interval = 0.01, ch1 = 1, ch2 = 2){
  w <- seq(start, end, by = interval)
  est_sd <- compute_spec(phi = phi, SIGMA = SIGMA,
                         w = w, P_max = P_max, ch1 = ch1, ch2 = ch2)
  return(est_sd)

}




##### compute quantile of spectral density ######
quantile_sd <- function(result, P, P_opt, n_t, sample.size = 1000L,
                        qt = c(0.025, 0.975), ch1 = 1, ch2 = 2){
  phi_fwd <- result$phi_fwd
  phi_bwd <- result$phi_bwd
  Cnt_fwd <- result$Cnt_fwd
  Cnt_bwd <- result$Cnt_bwd
  SIGMA <- result$St_fwd[[P_opt]][, , n_t - P]
  n_w <- length(seq(0.001, 0.499, 0.01))
  sample.sd <- rep(list(matrix(NA, nrow = n_w*n_t, sample.size)), 4)
  for(i in 1:sample.size){
    AR <- gen_AR_sample(phi_fwd = phi_fwd, phi_bwd = phi_bwd,
                        Cnt_fwd = Cnt_fwd, Cnt_bwd = Cnt_bwd,
                        n_I = 2, P_opt = P_opt, P_max = P, h = 0)
    sd <- sample_sd(AR[[P_opt]]$forward, SIGMA, P_max = P, ch1 = ch1, ch2 = ch2)
    sample.sd[[1]][, i] <- as.vector(sd[[1]])
    sample.sd[[2]][, i] <- as.vector(sd[[2]])
    sample.sd[[3]][, i] <- as.vector(sd[[3]])
    sample.sd[[4]][, i] <- as.vector(sd[[4]])
  }
  sd_lb <- lapply(sample.sd, function(x) apply(x, 1, quantile, qt[1]))
  sd_lb <- lapply(sd_lb, function(x) matrix(x, nrow = n_t, ncol = n_w))
  sd_ub <- lapply(sample.sd, function(x) apply(x, 1, quantile, qt[2]))
  sd_ub <- lapply(sd_ub, function(x) matrix(x, nrow = n_t, ncol = n_w))
  return(list("lb" = sd_lb, "ub" = sd_ub))
}


### computed smoothed function
compute_sm_fc <- function(ar, y, P_opt, h, P){
  n_I <- dim(y)[1]
  n_t <- dim(y)[2] - h
  y_sm <- matrix(nrow = n_I, ncol = n_t + h)
  for(i in (P+1):(n_t+h)){
    y_tmp <- rep(0, n_I)
    for(j in 1:P_opt){
      ar_tmp <- matrix(ar[, i, j], n_I, n_I, byrow = TRUE)
      y_tmp <- y_tmp + ar_tmp%*% as.matrix(ifelse(rep(i-j, n_I)>rep(n_t, n_I), y_sm[, i-j], y[, i-j]))
    }
    y_sm[, i] <- y_tmp
  }
  y_sm[, 1:P] <- y[, 1:P]
  return(y_sm)
}


## forecast function
gen_fc <- function(result, h, P_opt,
                   P_max, y, n_sample = 1000L,
                   qt = c(0.025, 0.5, 0.975), uncertainty = TRUE){
  ## initialization
  phi_fwd <- result$phi_fwd
  phi_bwd <- result$phi_bwd
  Ct_fwd <- result$Cnt_fwd
  Ct_bwd <- result$Cnt_bwd
  St_fwd <- result$St_fwd[[P_opt]]
  delta_fwd <- result$delta_fwd
  delta_bwd <- result$delta_bwd
  n_t <- dim(St_fwd)[3]
  n_I <- dim(St_fwd)[1]
  n_I2 <- n_I*n_I
  St_T <- St_fwd[, , n_t - P]
  phi_fwd_opt <- phi_fwd[, , 1:P_opt, drop= FALSE]
  phi_bwd_opt <- phi_bwd[, , 1:P_opt, drop= FALSE]
  phi_fwd_fc <- array(dim = c(n_I2, n_t+h, P_opt))
  phi_bwd_fc <- array(dim = c(n_I2, n_t+h, P_opt))
  Ct_fwd_fc <- rep(list(rep(list(NA), n_t+h)), P_opt)
  Ct_bwd_fc <- rep(list(rep(list(NA), n_t+h)), P_opt)
  y_fc <- matrix(nrow = n_I, ncol = n_t+h)
  error_fc <- rep(list(matrix(nrow = n_I, ncol = h)), 4)

  ## get mean of the forecast part
  y_fc[, 1:(n_t)] <- y[, 1:(n_t)]
  for(i in 1:P_opt){
    phi_fwd_fc[, 1:(n_t - P), i] <- phi_fwd_opt[, 1:(n_t - P), i]
    phi_bwd_fc[, 1:(n_t - P), i] <- phi_bwd_opt[, 1:(n_t - P), i]
    phi_fwd_fc[, (n_t+1):(n_t+h), i] <- rep(phi_fwd_opt[, n_t, i], h)
    phi_bwd_fc[, (n_t-P+1):(n_t+h), i] <- rep(phi_bwd_opt[, n_t-P, i], h + P)
    for(j in (P+1):(n_t)){
      Ct_fwd_fc[[i]][[j]] <- Ct_fwd[[i]][[j]]
      if(j <= n_t-P){
        Ct_bwd_fc[[i]][[j]] <- Ct_bwd[[i]][[j]]
      }else{
        Ct_bwd_fc[[i]][[j]] <- Ct_fwd[[i]][[j]]
      }
    }
    tmp_dCd <- diag(delta_fwd[i, ])%*%Ct_fwd[[i]][[n_t]]%*%diag(delta_fwd[i, ])
    for(j in 1:(h)){
      tmp <- (1-j)*Ct_fwd[[i]][[n_t]] + j*tmp_dCd
      Ct_fwd_fc[[i]][[(n_t+j)]] <- 0.5*tmp + 0.5*t(tmp)
      Ct_bwd_fc[[i]][[(n_t+j)]] <- Ct_fwd_fc[[i]][[n_t+j]]
    }
  }
  ar_tmp <- PAR_to_AR_fun(phi_fwd_fc, phi_bwd_fc, n_I)
  ar <- ar_tmp[[P_opt]]$forward
  if(uncertainty){
    for(i in 1:h){
      index_i <- n_t + i
      y_tmp <- rep(0, n_I)
      tmp_sample <- rmvn(n = n_sample, mu = rep(0, n_I), sigma = St_T)
      error_fc_tmp <- apply(tmp_sample, 2, quantile, qt)
      error_fc[[1]][, i] <- error_fc_tmp[1, ]
      error_fc[[2]][, i] <- error_fc_tmp[2, ]
      error_fc[[3]][, i] <- error_fc_tmp[3, ]
      error_fc[[4]][, i] <- apply(tmp_sample, 2, mean)
    }
  }

  return(list(ar = ar, error = error_fc, phi_fwd = phi_fwd_fc,
              phi_bwd = phi_bwd_fc, Ct_fwd = Ct_fwd_fc, Ct_bwd = Ct_bwd_fc))
}


compute_mse <- function(y_pred, y){
  return(mean((y_pred - y)^2))
}
