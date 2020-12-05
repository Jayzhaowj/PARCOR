####################################################
########## draw spectral density graph #############
####################################################

### phi is AR coefficient matrix stored in array, the slice of array is the each layer of the AR coefficients
### SIGMA is the innovation variance covariance matrix
### eeg is boolean value. If draw the plot of EEG data, then eeg = TRUE
### dir_name is the name of the plot
### P_max is the potential order


draw.density <- function(phi, SIGMA, start = 0.001, end = 0.499, interval = 0.01,
                         P_max, n_t, target_dir, dir_name, ch1 = 1, ch2 = 2, 
                         which.type = "all", plot = FALSE, xlab = 'time', est = TRUE,
                         ylab = 'frequency', eeg = FALSE, wind = FALSE, mice = FALSE,
                         time_depend = TRUE, ...){
  ## which.type is string control which type of plot to plot. 
  ## which.type can be 'all', 'sd1', 'sd2', 'coh' and 'pcoh', 
  ## 'all': all plots, 
  ## 'sd1': spectral density of the first process, 
  ## 'sd2': spectral density of the second process,
  ## 'coh': coherence between two processes,
  ## 'pcoh': partial coherence between two processes.
  ## plot = FALSE, it will not plot
  w <- seq(start, end, by = interval) 
  P <- dim(phi)[3]
  sd <- compute_spec(phi = phi, SIGMA = SIGMA, w = w, P_max = P_max, ch1 = ch1, ch2 = ch2, time_depend)
  if(plot){
    if(eeg){
      channel_names <- c('F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz', 'F4', 'C4', 'P4')
      constant1 <- 83.72/3600
      x_coord <- seq(constant1*P, 83.72 - constant1*P, length.out = n_t - 2 * P)
      constant2 <- 3600/(256/5)
      y_coord <- constant2 * w 
      main1 <- paste0("log SD: ", channel_names[ch1])
      main2 <- paste0("log SD: ", channel_names[ch2])
      main3 <- paste0("SC: ", channel_names[ch1], " and ", channel_names[ch2])
      main4 <- paste0("SPC: ", channel_names[ch1], " and ", channel_names[ch2])
      main5 <- bquote("PDC: "*.(channel_names[ch2])%->%.(channel_names[ch1]))
      main6 <- bquote("DTF: "*.(channel_names[ch2])%->%.(channel_names[ch1]))
    }else if(wind){
      x_coord <- seq(0, 1, length.out = n_t - 2*P)
      y_coord <- w 
      names <- list(bquote('M'['X']), bquote('M'['Y']), 
                 bquote('S'['X']), bquote('S'['Y']),
                 bquote('W'['X']), bquote('W'['Y']))
      main1 <- bquote(hat('g')[.(names[[ch1]])]*'(t, '*omega*')')
      main2 <- bquote(hat('g')[.(names[[ch2]])]*'(t, '*omega*')')
      main3 <- bquote(hat(rho)[.(names[[ch1]])*','*.(names[[ch2]])]*'(t, '*omega*')')
      main4 <- bquote(hat(gamma)[.(names[[ch1]])*','*.(names[[ch2]])]*'(t, '*omega*')')
      main5 <- bquote("PDC: "*.(names[[ch2]])%->%.(names[[ch1]]))
      main6 <- bquote("DTF: "*.(names[[ch2]])%->%.(names[[ch1]]))
    }else if(mice){
      x_coord <- seq(0, 1, length.out = n_t - 2*P)
      y_coord <- w 
      allen_labels <- c("SSp-ll.L","RSP.R","VISli.L","MOp.L","VISrl.R","VISal.L","SSp-n.L","SSs.L","SSp-n.R","SSp-m.R","TEa.L",    
                        "SSp-bfd.L", "VISl.R","AUD.L","VISrl.L","VISal.R","SSp-m.L","VISpor.R","SSp-tr.R","MOp.R","AUD.R", "VISam.R" , 
                        "VISp.L", "SSp-ul.L", "TEa.R","MOs.R","SSs.R","SSp-bfd.R" ,"VISpl.R","VISpm.R" ,"MOs.L","VISli.R","SSp-un.R" ,
                        "SSp-tr.L",  "VISa-L",    "VISpm.L",   "SSp-ul.R",  "SSp-ll.R",  "RSP.L", "SSp-un.L","VISpl.L","VISam.L", "VISl.L" ,
                        "VISp.R", "VISpor.L" ) 

      main1 <- paste0("log SD: ", allen_labels[ch1])
      main2 <- paste0("log SD: ", allen_labels[ch2])
      main3 <- paste0("SC: ", allen_labels[ch1], "&", allen_labels[ch2])
      main4 <- paste0("SPC: ", allen_labels[ch1], "&", allen_labels[ch2])
      main5 <- bquote("PDC: "*.(allen_labels[ch2])%->%.(allen_labels[ch1]))
      main6 <- bquote("DTF: "*.(allen_labels[ch2])%->%.(allen_labels[ch1]))
    }else{
      x_coord <- seq(0, 1, length.out = n_t - 2*P)
      y_coord <- w 
      if(est){
        main1 <- bquote(hat('g')[.(ch1)*','*.(ch1)]*'(t, '*omega*')')
        main2 <- bquote(hat('g')[.(ch2)*','*.(ch2)]*'(t, '*omega*')')
        main3 <- bquote(hat(rho)[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
        main4 <- bquote(hat(gamma)[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
        main5 <- bquote(hat("PDC")[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
        main6 <- bquote(hat("DTF")[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
      }else{
        main1 <- bquote('g'[.(ch1)*','*.(ch1)]*'(t, '*omega*')')
        main2 <- bquote('g'[.(ch2)*','*.(ch2)]*'(t, '*omega*')')
        main3 <- bquote(rho[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
        main4 <- bquote(gamma[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
        main5 <- bquote("PDC"[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
        main6 <- bquote("DTF"[.(ch1)*','*.(ch2)]*'(t, '*omega*')')
      }
      
    }
    if(time_depend){
      if(which.type == "sd1"| which.type == "all"){
        png(paste0(target_dir, dir_name, 'sd1.png'))
        par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                         "yellow", "#FF7F00", "red", "#7F0000"))
        tmp <- sd[[1]][(P+1):(n_t - P),  ]
        filled.contour(x_coord, y_coord, tmp, main = main1, 
                       color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        dev.off()
      }
      if(which.type == "sd2" | which.type == "all"){
        png(paste0(target_dir, dir_name, 'sd2.png'))
        par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                         "yellow", "#FF7F00", "red", "#7F0000"))
        tmp <- sd[[2]][(P+1):(n_t - P),  ]
        filled.contour(x_coord, y_coord, tmp, main = main2, 
                       color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        dev.off()
      }
      if(which.type == "coh" | which.type == "all"){
        png(paste0(target_dir, dir_name, 'coh.png'))
        par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                         "yellow", "#FF7F00", "red", "#7F0000"))
        filled.contour(x_coord, y_coord, sd[[3]][(P+1):(n_t - P),  ], xlab = xlab, ylab = ylab,
                       main = main3, color.palette = jet.colors, zlim = c(0, 1), ...)
        
        cat('Graph has been drawn!\n')
        dev.off()
      }
      if(which.type == "pcoh" | which.type == "all"){
        png(paste0(target_dir, dir_name, 'pcoh.png'))
        par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                         "yellow", "#FF7F00", "red", "#7F0000"))
        filled.contour(x_coord, y_coord, sd[[4]][(P+1):(n_t - P),  ], 
                       main = main4, xlab = xlab, ylab = ylab,
                       color.palette = jet.colors, zlim = c(0, 1), ...)
        cat('Graph has been drawn!\n')
        dev.off()
      }
      if(which.type == "pdc" | which.type == "all"){
        png(paste0(target_dir, dir_name, 'pdc.png'))
        par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                         "yellow", "#FF7F00", "red", "#7F0000"))
        filled.contour(x_coord, y_coord, sd[[5]][(P+1):(n_t - P),  ], 
                       main = main5, xlab = xlab, ylab = ylab,
                       color.palette = jet.colors, zlim = c(0, 1), ...)
        cat('Graph has been drawn!\n')
        dev.off()
      }
      if(which.type == "dtf" | which.type == "all"){
        png(paste0(target_dir, dir_name, 'dtf.png'))
        par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                         "yellow", "#FF7F00", "red", "#7F0000"))
        filled.contour(x_coord, y_coord, sd[[6]][(P+1):(n_t - P),  ], 
                       main = main6, xlab = xlab, ylab = ylab,
                       color.palette = jet.colors, zlim = c(0, 1), ...)
        cat('Graph has been drawn!\n')
        dev.off()
      }
    }else{
      if(which.type == "sd1"| which.type == "all"){
        #png(paste0(target_dir, dir_name, 'sd1.png'))
        #par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        #jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
        #                                 "yellow", "#FF7F00", "red", "#7F0000"))
        
        #filled.contour(x_coord, y_coord, sd[(P+1):(n_t - P), , 1], main = main1, 
        #               color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        plot(y_coord, sd[[1]], xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        #dev.off()
      }
      if(which.type == "sd2" | which.type == "all"){
        #png(paste0(target_dir, dir_name, 'sd2.png'))
        #par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        #jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
        #                                 "yellow", "#FF7F00", "red", "#7F0000"))
        #filled.contour(x_coord, y_coord, sd[(P+1):(n_t - P), , 2], main = main2, 
        #               color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        plot(y_coord, sd[[2]], xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        #dev.off()
      }
      if(which.type == "coh" | which.type == "all"){
        #png(paste0(target_dir, dir_name, 'sd2.png'))
        #par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        #jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
        #                                 "yellow", "#FF7F00", "red", "#7F0000"))
        #filled.contour(x_coord, y_coord, sd[(P+1):(n_t - P), , 2], main = main2, 
        #               color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        plot(y_coord, sd[[3]], xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        #dev.off()
      }
      if(which.type == "pcoh" | which.type == "all"){
        #png(paste0(target_dir, dir_name, 'sd2.png'))
        #par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        #jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
        #                                 "yellow", "#FF7F00", "red", "#7F0000"))
        #filled.contour(x_coord, y_coord, sd[(P+1):(n_t - P), , 2], main = main2, 
        #               color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        plot(y_coord, sd[[4]], xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        #dev.off()
      }
      if(which.type == "pdc" | which.type == "all"){
        #png(paste0(target_dir, dir_name, 'sd2.png'))
        #par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        #jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
        #                                 "yellow", "#FF7F00", "red", "#7F0000"))
        #filled.contour(x_coord, y_coord, sd[(P+1):(n_t - P), , 2], main = main2, 
        #               color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        plot(y_coord, sd[[5]], xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        #dev.off()
      }
      if(which.type == "dtf" | which.type == "all"){
        #png(paste0(target_dir, dir_name, 'sd2.png'))
        #par(par(mar=c(5,6,4,1)+.1), cex.lab = 2, cex.axis = 1.5, cex.main = 2.5)
        cat('Calculation has been completed! \n')
        #jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
        #                                 "yellow", "#FF7F00", "red", "#7F0000"))
        #filled.contour(x_coord, y_coord, sd[(P+1):(n_t - P), , 2], main = main2, 
        #               color.palette = jet.colors, xlab = xlab, ylab = ylab, ...)
        plot(y_coord, sd[[6]], xlab = xlab, ylab = ylab, ...)
        cat('Graph has been drawn!\n')
        #dev.off()
      }
    }
    
  }
  return(sd)
}
