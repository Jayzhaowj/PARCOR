
####### compute the ASE #########

comp.ase.cpp <- function(phi, SIGMA, phi_true, SIGMA_true, P_max, start = 0.001, end = 0.499, interval = 0.01){
  w <- seq(start, end, by = interval)
  n_t <- dim(phi)[2]
  ase <- c(0, 0, 0)
  TT <- (P_max+1):(n_t-P_max)
  true_sd <- compute_spec(phi = phi_true, SIGMA = SIGMA_true, w = w, P_max = P_max, ch1 = 1, ch2 = 2)
  est_sd <- compute_spec(phi = phi, SIGMA = SIGMA, w = w, P_max = P_max, ch1 = 1, ch2 = 2)
  ase[1] <- mean((est_sd[[1]][TT, ] - true_sd[[1]][TT, ])^2)
  ase[2] <- mean((est_sd[[2]][TT, ] - true_sd[[2]][TT, ])^2)
  ase[3] <- mean((est_sd[[3]][TT, ] - true_sd[[3]][TT, ])^2)
  return(ase)
}
