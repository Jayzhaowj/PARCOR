library(snowfall)


run_parcor_parallel <- function(F1, delta, P, sample_size = 200L, chains = 1, DIC = TRUE, S_0,
                                uncertainty = TRUE){

  n_t <- ncol(F1)
  n_I <- nrow(F1)
  n_I2 <- n_I^2
  n_delta <- nrow(delta[, , 1, drop = FALSE])
  F1_fwd <- F1
  F1_bwd <- F1
  phi_fwd <- array(NA, dim = c(n_I2, n_t, P))
  phi_bwd <- array(NA, dim = c(n_I2, n_t, P))
  St_fwd <- rep(list(NA), P)
  St_bwd <- rep(list(NA), P)
  Cnt_fwd <- rep(list(NA), P)
  Cnt_bwd <- rep(list(NA), P)
  ar_coef_prev <- list(NA)
  ar_coef <- rep(list(NA), P)
  delta_fwd <- matrix(NA, nrow = P, ncol = n_I2)
  delta_bwd <- matrix(NA, nrow = P, ncol = n_I2)
  val_DIC <- numeric(P)
  ll_DIC <- numeric(P)
  es_DIC <- numeric(P)
  ll_fwd <- numeric(P)
  ll_bwd <- numeric(P)
  ll <- matrix(NA, nrow = P, ncol = n_delta)
  sfInit(parallel = TRUE, cpus = 2, type = "SOCK")
  sfLibrary(PARCOR)
  sfExport("S_0", "delta", "P", "sample_size", "chains", "uncertainty")
  #browser()
  for(i in 1:P){
    sfExport("F1_fwd", "F1_bwd", "i")
    res <- sfLapply(x = c(1, 0), fun = function(x) filter_smooth(F1_fwd, F1_bwd, S_0, i, as.matrix(delta[, , i]),
                                                                 x, P, x, sample_size, chains, uncertainty))
    tmp_fwd <- res[[1]]
    tmp_bwd <- res[[2]]
    delta_fwd[i, ] <- tmp_fwd$delta_min
    delta_bwd[i, ] <- tmp_bwd$delta_min
    ll_DIC[i] <- tmp_fwd$ll_DIC
    es_DIC[i] <- tmp_fwd$ES_mean
    St_fwd[[i]] <- tmp_fwd$St
    St_bwd[[i]] <- tmp_bwd$St
    if(uncertainty){
      Cnt_fwd[[i]] <- tmp_fwd$Cnt
      Cnt_bwd[[i]] <- tmp_bwd$Cnt
    }
    F1_fwd <- tmp_fwd$F1new
    F1_bwd <- tmp_bwd$F1new
    phi_fwd[, , i] <- tmp_fwd$mnt
    phi_bwd[, , i] <- tmp_bwd$mnt
    ll_fwd[i] <- tmp_fwd$ll_max
    ll_bwd[i] <- tmp_bwd$ll_max
    ll[i, ] <- tmp_fwd$ll
    cat("iterations: ", i, "/", P, "\n")

  }
  ar_coef <- PAR_to_AR_fun(phi_fwd, phi_bwd, n_I)
  val_DIC <- 2*(cumsum(es_DIC) - ll_DIC)
  sfStop()
  return(list("phi_fwd" = phi_fwd,
              "phi_bwd" = phi_bwd,
              "St_fwd" = St_fwd,
              "St_bwd" = St_bwd,
              "Cnt_fwd" = Cnt_fwd,
              "Cnt_bwd" = Cnt_bwd,
              "ll_fwd" = ll_fwd,
              "ll_bwd" = ll_bwd,
              "delta_fwd" = delta_fwd,
              "delta_bwd" = delta_bwd,
              "ll" = ll,
              "ar_coef" = ar_coef,
              "DIC" = val_DIC,
              "ES_mean" = es_DIC,
              "ll_DIC" = ll_DIC,
              "F1_fwd" = F1_fwd,
              "F1_bwd" = F1_bwd))
}
