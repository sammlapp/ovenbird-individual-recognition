install.packages("ggeffects")
library(ggeffects)

count_data <- read.csv('../resources/per_point_counts_and_covars.csv')
# count_data$naive_occupancy <- count_data$X2021>0
count_data$Block <- as.factor(count_data$site)
count_data$year <- as.factor(count_data$year)
count_data$canopy2 <- count_data$canopy_cover_50m_avg^2

# standardize variables to mean 0, stdev 1
count_data$canopy_scaled <- as.numeric(scale(count_data$canopy_cover_50m_avg))
count_data$slope_scaled  <- as.numeric(scale(count_data$Slope))
count_data$ba_scaled     <- as.numeric(scale(count_data$Prism.BA..ft2.ac.))

# fit the full model with Block and Year effects plus 3 habitat covariates
full<-glm("n_individuals ~ Block + year + slope_scaled + canopy_scaled + ba_scaled",data=count_data,family=poisson)
ci<-confint(full)
summary(full)

null<-glm("n_individuals ~ Block + year",data=count_data,family=poisson)
summary(null)

# Extract coefficients with confidence intervals
coef_table_full <- as.data.frame(summary(full)$coefficients)
coef_table_full$Term <- rownames(coef_table_full)
rownames(coef_table_full) <- NULL
write.csv(coef_table_full,'../../results/abundance_model_parameter_estimates/glm_abundance_full.csv')

coef_table_null <- as.data.frame(summary(null)$coefficients)
coef_table_null$Term <- rownames(coef_table_null)
rownames(coef_table_null) <- NULL
write.csv(coef_table_null,'../../results/abundance_model_parameter_estimates/glm_abundance_null.csv')

# Model dredging with mulnm: 
# fit all subsets of variables from the Full model
# but we don't need to consider the models that leave out Block and Year effects
library(MuMIn)
options(na.action = "na.fail")
comparison <- dredge(full)
comparison
write.csv(data.frame(comparison),'../../results/abundance_model_parameter_estimates/abundance_model_comparison.csv')


## Plot Partial effects covariate relationships
# note that these effects are just the added partial effect of the covariate
# on the standardized scale. Interpretation is tricky, but back-transforming the 
# scale to the original value makes it easy to mis-interpret as a marginal effect
covariates <- c("slope_scaled", "canopy_scaled", "ba_scaled")

plots <- lapply(covariates, function(var) {
  ggpredict(full, terms = var) %>%
    plot()+
    labs(title = NULL)
})

# Show all together (if using patchwork or cowplot)
library(patchwork)
wrap_plots(plots)
# save
ggsave("../../figures/abundance_partial_effects.pdf",height=2,width=6.5)
