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
# hierarchical CJS model with random site and fixed year and block effects
# adapted from Applied Hierarchical Modeling Vol2 3.4.2  

# NOTE: use "set working directory -> to source file location"

# load detection history and point covariate table
oven_data <- read.csv("../resources/nimble_detection_histories_and_covariates.csv")
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


# load point-level habitat and landscape covariates
site_covars <-read.csv("../resources/nimble_site_habitat_covariates.csv")
canopy_scaled <- standardize(site_covars$canopy_cover_50m_avg)
slope_scaled <- standardize(site_covars$Slope)
ba_scaled <-standardize(site_covars$Prism.BA..ft2.ac.)
# indicator variable: 1 for SGL034 block, 0 for Ohiopyple block
blockSGL034 <- as.numeric(site_covars$site=="SGL034")

# Bundle and summarize data set
str(bdata <- list(MARR = MARR, R = R, n.site = nsite, n.occ = nyear,
    ba_scaled = ba_scaled, slope_scaled = slope_scaled, canopy_scaled = canopy_scaled, blockSGL034=blockSGL034))


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

write.csv(full_model$summary,"../../results/survival_model_parameter_estimates/full_cjs_with_point_covariates.csv")

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
ggsave("../../figures/parameter_posteriors/full_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)

#############################################

## NULL model equivalent blocking variables: fixed offset on year and block; random point

# Initial values
inits <- function(){list()}

# Parameters monitored
params <- c("phi_year", "p_intercept", "sd.lphi.site", "phi_ohiopyle_2021", "phi_sgl034_2021")

# Call JAGS, check convergence and summarize posteriors
null_model <- jags(bdata, inits, params, "cjs_null_random_site_effect_fixed_yr_effect.txt", n.adapt = na, n.chains = nc,
             n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)

write.csv(null_model$summary,"../../results/survival_model_parameter_estimates/null_cjs_with_year_and_point_effects.csv")

plots <-create_posterior_plots(null_model)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("../../figures/parameter_posteriors/null_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)


## Basic CJS model: no year or block effects, ignore nesting of individuals within points
inits <- function(){list()}

# Parameters monitored
params <- c("phi_intercept","p_intercept")

# Call JAGS, check convergence and summarize posteriors
cjs <- jags(bdata, inits, params, "cjs_base.txt", n.adapt = na, n.chains = nc,
                   n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
cjs$summary
write.csv(cjs$summary,"../../results/survival_model_parameter_estimates/base_cjs.csv")

plots <-create_posterior_plots(cjs)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("../../figures/parameter_posteriors/base_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 3)

###############################################################

### Repeat all 3 models with detection history of "residents" only
# this detection history only marks a bird as present in a year
# if it is detected in at least 4 of 12 days in the 12 consecutive day detection history

# load my data
oven_resident_data <- read.csv("../resources/nimble_detection_histories_and_covariates_residents.csv")
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
write.csv(full_model$summary,"../../results/survival_model_parameter_estimates/residents_only_full_cjs_with_point_covariates.csv")
plots <-create_posterior_plots(full_model)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("../../figures/parameter_posteriors/residents_only_full_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)

# Fit null model with site random effect, block effect, year effect on phi

# Initial values
inits <- function(){list()}
# Parameters monitored
params <- c("phi_year", "p_intercept", "sd.lphi.site", "phi_ohiopyle_2021", "phi_sgl034_2021")
# Call JAGS, check convergence and summarize posteriors
null_model <- jags(bdata, inits, params, "cjs_null_random_site_effect_fixed_yr_effect.txt", n.adapt = na, n.chains = nc,
                   n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
write.csv(null_model$summary,"../../results/survival_model_parameter_estimates/residents_only_null_cjs_with_year_and_point_effects.csv")
plots <-create_posterior_plots(null_model)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("../../figures/parameter_posteriors/residents_only_null_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 8)


# Fit basic CJS model with residents-only detection history
inits <- function(){list()}
params <- c("phi_intercept","p_intercept")
cjs <- jags(bdata, inits, params, "cjs_base.txt", n.adapt = na, n.chains = nc,
            n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)
write.csv(cjs$summary,"../../results/survival_model_parameter_estimates/residents_only_base_cjs.csv")
plots <-create_posterior_plots(cjs)
wrap_plots(plots, ncol = 3) #displays the plot
ggsave("../../figures/parameter_posteriors/residents_only_base_cjs_posterior_plots.pdf", wrap_plots(plots, ncol = 3), device = cairo_pdf, width = 12, height = 3)



