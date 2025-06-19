#adapted from code for
#   Applied hierarchical modeling in ecology
#   Modeling distribution, abundance and species richness using R and BUGS
#   Volume 2: Dynamic and Advanced models
#   Marc Kéry & J. Andy Royle
# Chapter 3 : HIERARCHICAL MODELS OF SURVIVAL
install.packages("AHMbook")
install.packages("jagsUI")
install.packages("patchwork")
library(jagsUI)
library(AHMbook)
library(ggplot2)
library(dplyr)
library(patchwork)
library(rjags)
library(tidyr)
# Spatial hierarchical CJS models
# ===================================

# load my data
oven_data <- read.csv("./nimble_detection_histories_and_covariates.csv")
# extract yearly detection history
ch <- as.matrix(oven_data[,1:4]) # columns 3-6 correspond to yearly detections
sitevec <- oven_data$point_numeric_id

# Report number of captures per bird, year and site
table(apply(ch, 1, sum))    # Table of capture frequency per bird
apply(ch, 2, sum)           # Number of birds per year
plot(table(table(sitevec))) # Frequency distribution of number of
                            # birds captured per site (not shown)
summary(as.numeric(table(sitevec)))
(nyear <- ncol(ch))        # Number years: 4
(marr <- ch2marray(ch))

# Calculate the number of birds released each year
(r <- apply(marr, 1, sum))

# Create 3d (or multi-site) m-array and r array: MARR and R
(nsite <- length(unique(sitevec)) )
n_recapture_years <- nyear-1
MARR <- array(NA, dim = c(n_recapture_years, nyear, nsite))
R <- array(NA, dim = c(n_recapture_years, nsite))
for(k in 1:nsite){
  sel.part <- ch[sitevec == k,, drop = FALSE]
  ma <- ch2marray(sel.part)
  MARR[,,k] <- ma
  R[,k] <- apply(ma, 1, sum)
}
MARR ; R 

# hierarchical CJS model with random site and fixed year and block effects
# adapted from Applied Hierarchical Modeling Vol2 3.4.2  

# load point-level habitat and landscape covariates
site_covars <-read.csv("./nimble_site_habitat_covariates.csv")
canopy_scaled <- standardize(site_covars$canopy_cover_50m_avg)
slope_scaled <- standardize(site_covars$Slope)
ba_scaled <-standardize(site_covars$Prism.BA..ft2.ac.)
# indicator variable: 1 for SGL034 block, 0 for Ohiopyple block
blockSGL034 <- as.numeric(site_covars$site=="SGL034")

# Bundle and summarize data set
str(bdata <- list(MARR = MARR, R = R, n.site = nsite, n.occ = nyear,
    ba_scaled = ba_scaled, slope_scaled = slope_scaled, canopy_scaled = canopy_scaled, blockSGL034=blockSGL034))

# Specify model in BUGS language
# CJS with random effect on site-level survival, fixed effect on yearly survival and "block",
# and site-level covariates on survival
cat(file = "cjs_full_random_site_effect_fixed_yr_effect_covars.txt","
model {

  # Priors and linear models
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      
      # phi: survival at site s, time t
      phi[t, s] <- ilogit(lphi[t, s]) # survival
      
      # p: recapture probability given return
      # here, it is a constant across year t and site s
      p[t, s] <- p_intercept # p_year[t]
      
      # survival has a random effect for site, fixed effect for year, fixed effect for block
      # mean_lphi_site includes 3 covariates and an intercept
      lphi[t, s] <- mean_lphi_site[s] + lphi_site_random_effect[s] + lphi_block[s] + lphi_year[t]  # + lphi_intercept #
      
    }

    # Linear model for site-level effects on survival
    lphi_site_random_effect[s] ~ dnorm(0, tau.lphi.site)
    lphi_block[s] <- blockSGL034[s] * lphi_block_sgl034 # dummy variable for Block = SGL034, 0 if other block (Ohiopyle)
    mean_lphi_site[s] <- beta_slope * slope_scaled[s] + beta_ba * ba_scaled[s] + beta_canopy * canopy_scaled[s] # lphi_intercept (included in year fixed effect)

    # backtransform site and year means
    mean.phi.site[s] <- ilogit(mean_lphi_site[s])
    
  }
  
  # Hyperpriors for hyperparams
  
  # fixed effect for yearly survival and recapture
  # phi_intercept ~ dunif(0,1)
  # lphi_intercept <- logit(phi_intercept)
  
  # model probability of detection as constant across sites, years, and block
  p_intercept ~ dunif(0,1)
  
  for (t in 1:(n.occ-1)){
    
    phi_year[t] ~ dunif(0,1)
    lphi_year[t]<-logit(phi_year[t])
    
    #p_year[t] ~ dunif(0,1)
  }
  
  # fixed effect for block
  lphi_block_sgl034~dnorm(0,0.1)
  
  # backtransform site means for first year (2021)
  phi_sgl034_2021 <- ilogit(lphi_year[1]+lphi_block_sgl034)
  phi_ohiopyle_2021 <- ilogit(lphi_year[1])

  
  # survival mean (year 1)
  # mean.phi ~ dunif(0, 1)
  # lphi_intercept <- logit(mean.phi) # intercept accounted for in fixed year effects
  
  # survival random effect variance:
  # we use tau to represent the stdev of the logit-normal,
  # which relates to the sd of the normal distribution as tau=sd^-2
  sd.lphi.site ~ dunif(0, 3)
  tau.lphi.site <- pow(sd.lphi.site, -2)

  # calculate +/- 2 stdev of site survival
  phi.site.plus2sd <- ilogit(lphi_year[1]+2*sd.lphi.site)
  phi.site.minus2sd <- ilogit(lphi_year[1]-2*sd.lphi.site)

  # Coefficients for covariates
  beta_slope ~ dnorm(0, 0.1)
  beta_ba ~ dnorm(0, 0.1)
  beta_canopy ~ dnorm(0, 0.1)

  # Multinomial likelihood for the m-array data (JAGS style)
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      MARR[t,1:n.occ,s] ~ dmulti(pr[t, , s], R[t,s])
    }
  }

  # Define the cell probabilities of the m-array
  # Main diagonal
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      q[t,s] <- 1-p[t,s] # Probability of non-recapture
      pr[t,t,s] <- phi[t,s]*p[t,s]
      # Above main diagonal
      for (j in (t+1):(n.occ-1)){
        pr[t,j,s] <- prod(phi[t:j,s])*prod(q[t:(j-1),s])*p[j,s]
      } #j
      # Below main diagonal
      for (j in 1:(t-1)){
        pr[t,j,s] <- 0
      }
    }
  }

  # Last column of m-array: probability of non-recapture
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      pr[t,n.occ,s] <- 1-sum(pr[t,1:(n.occ-1),s])
    }
  }
}
")

# Initial values
inits <- function(){list()}

# Parameters monitored
params <- c("phi_year", "p_intercept", "sd.lphi.site", "beta_slope", "beta_ba", "beta_canopy", "phi_ohiopyle_2021", "phi_sgl034_2021")

# MCMC settings
na <- 5000 ; ni <- 60000 ; nt <- 30 ; nb <- 30000 ; nc <- 3
# na <- 5000 ; ni <- 6000 ; nt <- 3 ; nb <- 3000 ; nc <- 3 # faster, for testing

# Call JAGS check convergence and summarize posteriors
full_model <- jags(bdata, inits, params, "cjs_full_random_site_effect_fixed_yr_effect_covars.txt", n.adapt = na, n.chains = nc,
    n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)

write.csv(full_model$summary,"full_cjs_with_point_covariates.csv")

create_posterior_plots<-function (model){
  # Extract MCMC samples from all chains
  samples <- model$samples
  
  # Get parameter names from the first chain
  param_names <- colnames(samples[[1]])
  
  # Combine all chains into a single long data frame
  all_samples <- do.call(rbind, lapply(seq_along(samples), function(i) {
    as.data.frame(samples[[i]]) |>
      dplyr::mutate(chain = i)
  }))
  
  # Melt to long format
  long_samples <- pivot_longer(all_samples, 
                               cols = all_of(param_names),
                               names_to = "parameter", 
                               values_to = "value")
  # 
  # # Create individual density plots
  # plots <- lapply(unique(long_samples$parameter), function(param) {
  #   ggplot(dplyr::filter(long_samples, parameter == param), aes(x = value)) +
  #     geom_histogram(aes(y = ..density..), bins = 50, fill = "#69b3a2", color = "white", alpha = 0.7) +
  #     geom_density(color = "#1f77b4", size = 1) +
  #     labs(title = param, x = NULL, y = "Density") +
  #     theme_minimal(base_size = 11) +
  #     theme(plot.title = element_text(size = 10, face = "bold"))
  # })
  
  plots <- lapply(unique(long_samples$parameter), function(param) {
    df <- dplyr::filter(long_samples, parameter == param)
    stats <- quantile(df$value, probs = c(0.05, 0.5, 0.95))
    mean_val <- mean(df$value)
    
    ggplot(df, aes(x = value)) +
      geom_histogram(aes(y = ..density..), bins = 50, fill = "#69b3a2", color = "white", alpha = 0.7) +
      geom_density(color = "#1f77b4", size = 1) +
      geom_vline(xintercept = stats[1], color = "#1f77b4", linetype = "dashed", size = 0.6) +
      geom_vline(xintercept = mean_val, color = "black", linetype = "solid", size = 0.8) +
      geom_vline(xintercept = stats[3], color = "#1f77b4", linetype = "dashed", size = 0.6) +
      annotate("text", x = stats[1], y = Inf, label = sprintf("%.2f", stats[1]),
               vjust = 2, hjust = 1, angle = 90, size = 3, color = "black") +
      annotate("text", x = mean_val, y = Inf, label = sprintf("%.2f", mean_val),
               vjust = 2, hjust = 1, angle = 90, size = 3, color = "black") +
      annotate("text", x = stats[3], y = Inf, label = sprintf("%.2f", stats[3]),
               vjust = 2, hjust = 1, angle = 90, size = 3, color = "black") +
      labs(title = param, x = NULL, y = "Density") +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(size = 10, face = "bold"))
  })
  
  return(plots)
}

plots <-create_posterior_plots(full_model)
# Arrange using patchwork (e.g., 3 columns per row)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("full_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)

#############################################

## NULL model equivalent blocking variables: fixed offset on year and block; random point

cat(file = "cjs_null_random_site_effect_fixed_yr_effect.txt","
model {

  # Priors and linear models
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      
      # phi: survival at site s, time t
      phi[t, s] <- ilogit(lphi[t, s]) # survival
      
      # p: recapture probability given return
      # here, it is a constant across year t and site s
      p[t, s] <- p_intercept # p_year[t]
      
      # survival has a random effect for site, fixed effect for year, fixed effect for block
      # mean_lphi_site for null model is intercept-only (comes from lphi_year: one intercept per year)
      # block effect: dummy variable 1 for SGL034, 0 for Ohiopyle
      lphi[t, s] <- lphi_site_random_effect[s] + lphi_block[s] + lphi_year[t]  # mean_lphi_site[s] + 
      
    }

    # Linear model for site-level effects on survival
    lphi_site_random_effect[s] ~ dnorm(0, tau.lphi.site)
    lphi_block[s] <- blockSGL034[s] * lphi_block_sgl034 # dummy variable for Block = SGL034, 0 if other block (Ohiopyle)
  }
  
  # Hyperpriors for hyperparams
  
  # fixed effect for yearly survival and recapture
  # phi_intercept ~ dunif(0,1)
  # lphi_intercept <- logit(phi_intercept)
  p_intercept ~ dunif(0,1)
  
  for (t in 1:(n.occ-1)){
    
    phi_year[t] ~ dunif(0,1)
    lphi_year[t]<-logit(phi_year[t])
    
    # we don't model separate yearly detection probabilities
    #p_year[t] ~ dunif(0,1)
  }
  
  # fixed effect for block
  lphi_block_sgl034~dnorm(0,0.1)

  # backtransform site means for first year (2021)
  phi_sgl034_2021 <- ilogit(lphi_year[1]+lphi_block_sgl034)
  phi_ohiopyle_2021 <- ilogit(lphi_year[1])

  # survival random effect variance:
  # we use tau to represent the stdev of the logit-normal,
  # which relates to the sd of the normal distribution as tau=sd^-2
  sd.lphi.site ~ dunif(0, 3)
  tau.lphi.site <- pow(sd.lphi.site, -2)
  
  # calculate +/- 2 stdev of site survival
  phi.site.plus2sd <- ilogit(lphi_year[1]+2*sd.lphi.site)
  phi.site.minus2sd <- ilogit(lphi_year[1]-2*sd.lphi.site)

  mean.p ~ dunif(0, 1)

  # Coefficients for covariates
  beta_slope ~ dnorm(0, 0.1)
  beta_ba ~ dnorm(0, 0.1)
  beta_canopy ~ dnorm(0, 0.1)

  # Multinomial likelihood for the m-array data (JAGS style)
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      MARR[t,1:n.occ,s] ~ dmulti(pr[t, , s], R[t,s])
    }
  }

  # Define the cell probabilities of the m-array
  # Main diagonal
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      q[t,s] <- 1-p[t,s] # Probability of non-recapture
      pr[t,t,s] <- phi[t,s]*p[t,s]
      # Above main diagonal
      for (j in (t+1):(n.occ-1)){
        pr[t,j,s] <- prod(phi[t:j,s])*prod(q[t:(j-1),s])*p[j,s]
      } #j
      # Below main diagonal
      for (j in 1:(t-1)){
        pr[t,j,s] <- 0
      }
    }
  }

  # Last column of m-array: probability of non-recapture
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      pr[t,n.occ,s] <- 1-sum(pr[t,1:(n.occ-1),s])
    }
  }
}
")


# Initial values
inits <- function(){list()}

# Parameters monitored
params <- c("phi_year", "p_intercept", "sd.lphi.site", "phi_ohiopyle_2021", "phi_sgl034_2021")
# , "phi.grid" # for predictions within JAGS

# Call JAGS, check convergence and summarize posteriors
null_model <- jags(bdata, inits, params, "cjs_null_random_site_effect_fixed_yr_effect.txt", n.adapt = na, n.chains = nc,
             n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)

write.csv(null_model$summary,"null_cjs_with_year_and_point_effects.csv")

plots <-create_posterior_plots(null_model)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("null_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)


## Simplest model: no year or block effects
cat(file = "cjs_base.txt","
model {

  # Priors and linear models
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      
      # phi: probability of survival and return
      # here it is constant across year t and site s
      phi[t, s] <- ilogit(lphi_intercept) # survival
      lphi[t, s] <- lphi_intercept
      
      # p: recapture probability given return
      # here, it is a constant across year t and site s
      p[t, s] <- p_intercept # p_year[t]
      
    }
  }
  
  # Hyperpriors for hyperparams
  
  # fixed effect for yearly survival and recapture
  p_intercept ~ dunif(0,1)
  phi_intercept ~ dunif(0,1)
  lphi_intercept<-logit(phi_intercept)

  # Multinomial likelihood for the m-array data (JAGS style)
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      MARR[t,1:n.occ,s] ~ dmulti(pr[t, , s], R[t,s])
    }
  }

  # Define the cell probabilities of the m-array

  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      # Main diagonal
      q[t,s] <- 1-p[t,s] # Probability of non-recapture
      pr[t,t,s] <- phi[t,s]*p[t,s]
      # Above main diagonal
      for (j in (t+1):(n.occ-1)){
        pr[t,j,s] <- prod(phi[t:j,s])*prod(q[t:(j-1),s])*p[j,s]
      } #j
      # Below main diagonal
      for (j in 1:(t-1)){
        pr[t,j,s] <- 0
      }
    }
  }

  # Last column of m-array: probability of non-recapture
  for (s in 1:n.site){
    for (t in 1:(n.occ-1)){
      pr[t,n.occ,s] <- 1-sum(pr[t,1:(n.occ-1),s])
    }
  }
}
")


# Initial values
inits <- function(){list()}

# Parameters monitored
params <- c("phi_intercept","p_intercept")

# Call JAGS, check convergence and summarize posteriors
cjs <- jags(bdata, inits, params, "cjs_base.txt", n.adapt = na, n.chains = nc,
                   n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
cjs$summary
write.csv(cjs$summary,"base_cjs.csv")

plots <-create_posterior_plots(cjs)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("base_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 3)


### Repeat with detection history of "residents" only
# this detection history only marks a bird as present in a year
# if it is detected in at least 4 of 12 days in the 12 consecutive day detection history

# load my data
oven_resident_data <- read.csv("./nimble_detection_histories_and_covariates_residents.csv")
# extract yearly detection history
ch <- as.matrix(oven_resident_data[,1:4]) # columns 3-6 correspond to yearly detections
sitevec <- oven_data$point_numeric_id

table(apply(ch, 1, sum))    # Table of capture frequency per bird
apply(ch, 2, sum)           # Number of birds per year
plot(table(table(sitevec))) # Frequency distribution of number of
# birds captured per site (not shown)
summary(as.numeric(table(sitevec)))
(nyear <- ncol(ch))        # Number years: 4
(marr <- ch2marray(ch))

# Calculate the number of birds released each year
(r <- apply(marr, 1, sum))

# Create 3d (or multi-site) m-array and r array: MARR and R
(nsite <- length(unique(sitevec)) )
n_recapture_years <- nyear-1
MARR <- array(NA, dim = c(n_recapture_years, nyear, nsite))
R <- array(NA, dim = c(n_recapture_years, nsite))
for(k in 1:nsite){
  sel.part <- ch[sitevec == k,, drop = FALSE]
  ma <- ch2marray(sel.part)
  MARR[,,k] <- ma
  R[,k] <- apply(ma, 1, sum)
}
MARR ; R 

# Bundle and summarize data set
str(bdata <- list(MARR = MARR, R = R, n.site = nsite, n.occ = nyear,
                  ba_scaled = ba_scaled, slope_scaled = slope_scaled, canopy_scaled = canopy_scaled, blockSGL034=blockSGL034))

# Initial values
inits <- function(){list()}
# Parameters monitored
params <- c("phi_year", "p_intercept", "sd.lphi.site", "beta_slope", "beta_ba", "beta_canopy", "phi_ohiopyle_2021", "phi_sgl034_2021")
# Call JAGS check convergence and summarize posteriors
full_model <- jags(bdata, inits, params, "cjs_full_random_site_effect_fixed_yr_effect_covars.txt", n.adapt = na, n.chains = nc,
                   n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
write.csv(full_model$summary,"residents_only_full_cjs_with_point_covariates.csv")
plots <-create_posterior_plots(full_model)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("residents_only_full_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)

# Fit null model with site random effect, block effect, year effect on phi

# Initial values
inits <- function(){list()}
# Parameters monitored
params <- c("phi_year", "p_intercept", "sd.lphi.site", "phi_ohiopyle_2021", "phi_sgl034_2021")
# Call JAGS, check convergence and summarize posteriors
null_model <- jags(bdata, inits, params, "cjs_null_random_site_effect_fixed_yr_effect.txt", n.adapt = na, n.chains = nc,
                   n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
write.csv(null_model$summary,"residents_only_null_cjs_with_year_and_point_effects.csv")
plots <-create_posterior_plots(null_model)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("residents_only_null_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)


# Fit basic CJS model with residents-only detection history
inits <- function(){list()}
params <- c("phi_intercept","p_intercept")
cjs <- jags(bdata, inits, params, "cjs_base.txt", n.adapt = na, n.chains = nc,
            n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
write.csv(cjs$summary,"residents_only_base_cjs.csv")
plots <-create_posterior_plots(cjs)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("residents_only_base_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 3)



