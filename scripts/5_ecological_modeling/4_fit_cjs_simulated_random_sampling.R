# Fit CJS models to estimate surivavl when using simulated detection histories
# simulations are of annotating a random number of songs per point per year
# to create individual detection histories
# we loop over each detection history (5 repetitions of each N samples)
# and save the apparent survival estimate phi and detection probability
# estimate p to a table
install.packages("jagsUI")
library(jagsUI)
library(stringr)

## Base CJS: constant phi and p across time and space

# Directory containing detection history files
det_hist_dir <- "../../results/detection_histories/det_hists_with_random_sampling"
# List all relevant files
det_hist_files <- list.files(
  path = det_hist_dir,
  pattern = "^random_\\d+_det_hist_\\d+\\.csv$",
  full.names = TRUE
)
det_hist_files = c(det_hist_files, "./det_hists_with_random_sampling/full_det_hist.csv" )



for (f in det_hist_files) {
  cat("Processing", f, "\n")
  
  # Extract a short identifier for output file naming
  file_base <- tools::file_path_sans_ext(basename(f))
  
  det_hist <- read.csv(f)
    
  ch <- as.matrix(det_hist[,2:5]) # columns 3-6 correspond to yearly detections
  sitevec <- rep(1,length(det_hist))
  
  # Report number of captures per bird, year and site
  table(apply(ch, 1, sum))    # Table of capture frequency per bird
  apply(ch, 2, sum)           # Number of birds per year
  nyear <- ncol(ch)    # Number years: 4
  marr <- ch2marray(ch)
  
  # Calculate the number of birds released each year
  r <- apply(marr, 1, sum)
  
  # Create 3d (or multi-site) m-array and r array: MARR and R
  nsite <- length(unique(sitevec))
  n_recapture_years <- nyear-1
  MARR <- array(NA, dim = c(n_recapture_years, nyear, nsite))
  R <- array(NA, dim = c(n_recapture_years, nsite))
  for(k in 1:nsite){
    sel.part <- ch[sitevec == k,, drop = FALSE]
    ma <- ch2marray(sel.part)
    MARR[,,k] <- ma
    R[,k] <- apply(ma, 1, sum)
  }
  
  # Initial values
  inits <- function(){list()}
  na <- 5000 ; ni <- 6000 ; nt <- 3 ; nb <- 3000 ; nc <- 3 
  
  # Parameters monitored
  params <- c("phi_intercept","p_intercept")
  
  # Bundle and summarize data set
  str(bdata <- list(MARR = MARR, R = R, n.site = nsite, n.occ = nyear))
  
  # Call JAGS, check convergence and summarize posteriors
  
  cjs <- jags(bdata, inits, params, "cjs_base.txt", n.adapt = na, n.chains = nc,
              n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
  
  cjs_summary <-as.data.frame(cjs$summary)
  
  # Extract n_reviewed and run from filename
  matches <- str_match(file_base, "random_(\\d+)_det_hist_(\\d+)")
  n_reviewed <- as.integer(matches[2])
  run <- as.integer(matches[3])
  
  # Extract phi and p
  phi <- cjs_summary$mean[1]
  p <- cjs_summary$mean[2]
  
  # Store results
  if (!exists("summary_df")) summary_df <- data.frame()
  summary_df <- rbind(summary_df, data.frame(n_reviewed = n_reviewed, run = run, phi = phi, p = p))
}

write.csv(summary_df, file.path(out_dir, "../../results/survival_model_parameter_estimates/simulated_random_sampling_cjs_summary_values.csv"), row.names = FALSE)

