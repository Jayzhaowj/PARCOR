

run_TVVAR2 <- function(x.sim, P_max, n_0, S_0, delta, C_0){
  n_t <- dim(x.sim)[2]
  n_I <- dim(x.sim)[1]
  n_I2 <- n_I ** 2
  G <- diag(n_I2)
  mk_0 <- matrix(rep(0, n_I2), ncol = 1)
  Ck_0 <- C_0 * diag(n_I2)
  aic <- rep(0, P_max)
  bic <- rep(0, P_max)
  delta_TVVAR <- delta[[1]]
  ll <- numeric(P_max)
  result <- rep(list(NA), P_max)
  result[[1]] <- filter_smooth_TVVAR(F1 = x.sim, G = G, mk_0 = mk_0, 
                                     Ck_0 = Ck_0, n_0 = n_0, S_0 = S_0, m = 1, 
                                     delta = delta_TVVAR, pp = P_max)
  for(i in 2:P_max){
    delta_TVVAR <- delta[[i]]
    # count the number of parameters
    num_par <- n_I2*i + i*n_I*(n_I + 1)/2
    G <- diag(n_I2 * i)
    mk_0 <- matrix(rep(0, n_I2 * i), ncol = 1)
    Ck_0 <- C_0 * diag(n_I2 * i)
    result[[i]] <- filter_smooth_TVVAR(F1 = x.sim, G = G, mk_0 = mk_0, 
                                       Ck_0 = Ck_0, n_0 = n_0, S_0 = S_0, m = i, 
                                       delta = delta_TVVAR, pp = P_max)
  }
  return(result)
}
