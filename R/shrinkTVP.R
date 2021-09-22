
mcmc_sPARCOR <- function(y_fwd,
                         y_bwd,
                         S_0,
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
                         c_tuning_par_xi = 1,
                         c_tuning_par_tau = 1,
                         display_progress = TRUE,
                         ret_beta_nc = FALSE,
                         ind,
                         skip){


  # Input checking ----------------------------------------------------------


  # default hyperparameter values
  default_hyper <- list(c0 = 2.5,
                         g0 = 5,
                         G0 = 5 / (2.5 - 1),
                         e1 = 0.001,
                         e2 = 0.001,
                         d1 = 0.001,
                         d2 = 0.001,
                         b_xi = 10,
                         b_tau = 10,
                         nu_xi = 5,
                         nu_tau = 5)

  # default sv params
  #default_hyper_sv <- list(Bsigma_sv = 1,
  #                          a0_sv = 5,
  #                          b0_sv = 1.5,
  #                          bmu = 0,
  #                          Bmu = 1)

  # Change hyperprior values if user overwrites them
  if (missing(hyperprior_param)){
    hyperprior_param <- default_hyper
  } else {

    # Check that hyperprior_param and sv_param are a list
    if (is.list(hyperprior_param) == FALSE | is.data.frame(hyperprior_param)){
      stop("hyperprior_param has to be a list")
    }

    stand_nam <- names(default_hyper)
    user_nam <- names(hyperprior_param)

    # Give out warning if an element of the parameter list is misnamed
    if (any(!user_nam %in% stand_nam)){
      wrong_nam <- user_nam[!user_nam %in% stand_nam]
      warning(paste0(paste(wrong_nam, collapse = ", "),
                     ifelse(length(wrong_nam) == 1, " has", " have"),
                     " been incorrectly named in hyperprior_param and will be ignored"),
              immediate. = TRUE)
    }

    # Merge users' and default values and ignore all misnamed values
    missing_param <- stand_nam[!stand_nam %in% user_nam]
    hyperprior_param[missing_param] <- default_hyper[missing_param]
    hyperprior_param <- hyperprior_param[stand_nam]
  }




  # Check if all numeric inputs are correct
  to_test_num <- list(lambda2 = lambda2,
                   kappa2 = kappa2,
                   a_xi = a_xi,
                   a_tau = a_tau,
                   c_tuning_par_xi = c_tuning_par_xi,
                   c_tuning_par_tau = c_tuning_par_tau)

  if (missing(hyperprior_param) == FALSE){
    to_test_num <- c(to_test_num, hyperprior_param)
  }



  if ((niter - nburn) < 2){
    stop("niter has to be larger than or equal to nburn + 2")
  }

  if (nthin == 0){
    stop("nthin can not be 0")
  }

  if ((niter - nburn)/2 < nthin){
    stop("nthin can not be larger than (niter - nburn)/2")
  }



  # # Check if formula is a formula
  # if (inherits(formula, "formula") == FALSE){
  #   stop("formula is not of class formula")
  # }




  # Formula interface -------------------------------------------------------


  #mf <- match.call(expand.dots = FALSE)
  #m <- match(x = c("formula", "data"), table = names(mf), nomatch = 0L)
  #mf <- mf[c(1L, m)]
  #mf$drop.unused.levels <- TRUE
  #mf$na.action <- na.pass
  #mf[[1L]] <- quote(stats::model.frame)
  #mf <- eval(expr = mf, envir = parent.frame())
  ## Create Vector y
  #y <- model.response(mf, "numeric")
  #mt <- attr(x = mf, which = "terms")
  ## Create Matrix X with dummies and transformations
  #x <- model.matrix(object = mt, data = mf)

  # # Check that there are no NAs in y and x
  # if (any(is.na(y))) {
  #   stop("No NA values are allowed in response variable")
  # }
  #
  # if (any(is.na(x))){
  #   stop("No NA values are allowed in covariates")
  # }
  #
  # colnames(x)[colnames(x) == "(Intercept)"] <- "Intercept"
  #
  # d <- dim(x)[2]
  n_I <- dim(y_fwd)[2]
  store_burn <- FALSE



  # Run sampler -------------------------------------------------------------


  runtime <- system.time({
    suppressWarnings({
      res <- do_mcmc_sPARCOR(y_fwd,
                          y_bwd,
                          S_0,
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
                          c_tuning_par_xi,
                          c_tuning_par_tau,
                          hyperprior_param$b_xi,
                          hyperprior_param$b_tau,
                          hyperprior_param$nu_xi,
                          hyperprior_param$nu_tau,
                          display_progress,
                          ret_beta_nc,
                          store_burn,
                          ind,
                          skip)
    })
  })

  # Throw an error if the sampler failed
  if (res$success_vals$success == FALSE){
    stop(paste0("The sampler failed at iteration ",
                res$success_vals$fail_iter,
                " while trying to ",
                res$success_vals$fail, ". ",
                "Try rerunning the model. ",
                "If the sampler fails again, try changing the prior to be more informative. ",
                "If the problem still persists, please contact the maintainer: ",
                maintainer("shrinkTVP")))
  } else {
    res$success_vals <- NULL
  }


  # Post process sampler results --------------------------------------------


  if (display_progress == TRUE){
    cat("Timing (elapsed): ", file = stderr())
    cat(runtime["elapsed"], file = stderr())
    cat(" seconds.\n", file = stderr())
    cat(round( (niter + nburn) / runtime[3]), "iterations per second.\n\n", file = stderr())
    cat("Converting results to coda objects and summarizing draws... ", file = stderr())
  }

  # Collapse sigma2 to single vector if sv=FALSE
  # if (sv == FALSE){
  #   res$sigma2f <- matrix(res$sigma2f[1, 1, ], ncol = 1)
  # }

  # Remove empty storage elements
  if (ret_beta_nc == FALSE){
    res[["betaf_nc"]] <- NULL
    res[["betab_nc"]] <- NULL
  }


  if (learn_kappa2 == FALSE){
    res[["kappa2f"]] <- NULL
    res[["kappa2b"]] <- NULL
  }

  if (learn_lambda2 == FALSE){
    res[["lambda2f"]] <- NULL
    res[["lambda2b"]] <- NULL
  }

  if (learn_a_xi == FALSE){
    res[["a_xif"]] <- NULL
    res[["a_xif_acceptance"]] <- NULL
    res[["a_xib"]] <- NULL
    res[["a_xib_acceptance"]] <- NULL
  }

  if (learn_a_tau == FALSE){
    res[["a_tauf"]] <- NULL
    res[["a_tauf_acceptance"]] <- NULL
    res[["a_taub"]] <- NULL
    res[["a_taub_acceptance"]] <- NULL
  }


  # Create object to hold prior values
  priorvals <- c()


  if (learn_a_tau == TRUE){
    priorvals["b_tau"] <- hyperprior_param$b_tau
    priorvals["nu_tau"] <- hyperprior_param$nu_tau
  } else {
    priorvals["a_tau"] <- a_tau
  }

  if (learn_a_xi == TRUE){
    priorvals["b_xi"] <- hyperprior_param$b_xi
    priorvals["nu_xi"] <- hyperprior_param$nu_xi
  } else {
    priorvals["a_xi"] <- a_xi
  }


  if (learn_lambda2 == TRUE){
    priorvals["e1"] <- hyperprior_param$e1
    priorvals["e2"] <- hyperprior_param$e2
  } else {
    priorvals["lambda2"] <- lambda2
  }

  if (learn_kappa2 == TRUE){
    priorvals["d1"] <- hyperprior_param$d1
    priorvals["d2"] <- hyperprior_param$d2
  } else {
    priorvals["kappa2"] <- kappa2
  }


  res$priorvals <- priorvals
#
#   # Add data to output
#   res[["model"]] <- list()
#   res$model$x <- x
#   res$model$y <- y
#   res$model$formula <- formula

  # res$summaries <- list()
  #
  # # add attributes to the individual if they are distributions or individual statistics
  # nsave <- ifelse(store_burn, floor(niter/nthin), floor((niter - nburn)/nthin))
  # for (i in names(res)){
  #
  #   attr(res[[i]], "type") <- ifelse(nsave %in% dim(res[[i]]), "sample", "stat")
  #
  #   # Name each individual sample for plotting frontend
  #   if (attr(res[[i]], "type") == "sample"){
  #
  #     if (dim(res[[i]])[2] == d){
  #
  #       colnames(res[[i]]) <- paste0(i, "_",  colnames(x))
  #
  #     } else if (dim(res[[i]])[2] == 2 * d){
  #
  #       colnames(res[[i]]) <- paste0(i, "_", rep( colnames(x), 2))
  #
  #     } else {
  #
  #       colnames(res[[i]]) <- i
  #
  #     }
  #   }
  #
  #   # Change objects to be coda compatible
  #   # Only apply to posterior samples
  #   if (attr(res[[i]], "type") == "sample"){
  #
  #     # Differentiate between TVP and non TVP
  #     if (is.na(dim(res[[i]])[3]) == FALSE){
  #
  #       # Create a sub list containing an mcmc object for each parameter in TVP case
  #       dat <- res[[i]]
  #       res[[i]] <- list()
  #       for (j in 1:dim(dat)[2]){
  #         res[[i]][[j]] <- as.mcmc(t(dat[, j, ]), start = niter - nburn, end = niter, thin = nthin)
  #         colnames(res[[i]][[j]]) <- paste0(i, "_", j, "_", 1:ncol(res[[i]][[j]]))
  #
  #         # make it of class mcmc.tvp for custom plotting function
  #         class(res[[i]][[j]]) <- c("mcmc.tvp", "mcmc")
  #
  #         attr(res[[i]][[j]], "type") <- "sample"
  #
  #         # Imbue each mcmc.tvp object with index
  #         attr(res[[i]][[j]], "index") <- zoo::index(y)
  #       }
  #
  #       if (length(res[[i]]) == 1){
  #         res[[i]] <- res[[i]][[j]]
  #         attr(res[[i]][[j]], "index") <- zoo::index(y)
  #       }
  #
  #       # Make it of type 'sample' again
  #       attr(res[[i]], "type") <- "sample"
  #
  #       # Rename
  #       if (dim(dat)[2] > 1){
  #         names(res[[i]]) <- colnames(dat)
  #       }
  #
  #
  #     } else {
  #
  #       res[[i]] <- as.mcmc(res[[i]], start = niter - nburn, end = niter, thin = nthin)
  #
  #     }
  #   }
  #
  #   # Create summary of posterior
  #   if (is.list(res[[i]]) == FALSE & attr(res[[i]], "type") == "sample"){
  #     if (i != "theta_sr" & !(i == "sigma2" & sv == TRUE) & i != "beta"){
  #       res$summaries[[i]] <- t(apply(res[[i]], 2, function(x){
  #         obj <- as.mcmc(x, start = niter - nburn, end = niter, thin = nthin)
  #         return(c("mean" = mean(obj),
  #                  "sd" = sd(obj),
  #                  "median" = median(obj),
  #                  "HPD" = HPDinterval(obj)[c(1, 2)],
  #                  "ESS" = effectiveSize(obj)))
  #       }))
  #     } else if (i == "theta_sr"){
  #       res$summaries[[i]] <- t(apply(res[[i]], 2, function(x){
  #         obj <- as.mcmc(abs(x), start = niter - nburn, end = niter, thin = nthin)
  #         return(c("mean" = mean(obj),
  #                  "sd" = sd(obj),
  #                  "median" = median(obj),
  #                  "HPD" = HPDinterval(obj)[c(1, 2)],
  #                  "ESS" = effectiveSize(obj)))
  #       }))
  #     }
  #   }
  # }
  #

  if (display_progress == TRUE) {
    cat("Done!\n", file = stderr())
  }

  # add some attributes for the methods and plotting
  attr(res, "class") <- "shrinkTVP"
  attr(res, "learn_a_xi") <- learn_a_xi
  attr(res, "learn_a_tau") <- learn_a_tau
  attr(res, "learn_kappa2") <- learn_kappa2
  attr(res, "learn_lambda2") <- learn_lambda2
  attr(res, "niter") <- niter
  attr(res, "nburn") <- nburn
  attr(res, "nthin") <- nthin

  #attr(res, "colnames") <-  colnames(x)
  attr(res, "index") <- zoo::index(y_fwd)



  return(res)
}
