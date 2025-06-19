install.packages("marginaleffects")


count_data <- read.csv('./counts_and_covars.csv')
# count_data$naive_occupancy <- count_data$X2021>0
count_data$Block <- as.factor(count_data$site)
count_data$year <- as.factor(count_data$year)
count_data$canopy2 <- count_data$canopy_cover_50m_avg^2

# Ovenbird habitat covars: from Birds of the World
# canopy height (don't have it), percent canopy cover
# LL depth, prey availability (don't have)
# percent ground cover(unclear what this means), slope, and basal area
# rescale vars
# count_data$canopy_scaled <- scale(count_data$canopy_cover_50m_avg)
# # count_data$canopy2_scaled <- scale(count_data$canopy2)
# # count_data$edge_dist_scaled <- scale(count_data$distance_to_forest_edge_m)
# count_data$slope_scaled <-scale(count_data$Slope)
# # count_data$ll_scaled <-scale(count_data$ll)
# count_data$ba_scaled <-scale(count_data$Prism.BA..ft2.ac.)

count_data$canopy_scaled <- as.numeric(scale(count_data$canopy_cover_50m_avg))
count_data$slope_scaled  <- as.numeric(scale(count_data$Slope))
count_data$ba_scaled     <- as.numeric(scale(count_data$Prism.BA..ft2.ac.))


# point replication is only for survival model no random effect on point
# (no multiple observations per point - year )

full<-glm("n_individuals ~ Block + year + slope_scaled + canopy_scaled + ba_scaled",data=count_data,family=poisson)
ci<-confint(full)
ci
summary(full)

null<-glm("n_individuals ~ Block + year",data=count_data,family=poisson)
summary(null)



# Compare model AIC
aic_table <- data.frame(
  Model = c("Full", "Null"),
  AIC = c(AIC(full), AIC(null))
)

# Extract coefficients with confidence intervals
coef_table_full <- as.data.frame(summary(full)$coefficients)
coef_table_full$Term <- rownames(coef_table_full)
rownames(coef_table_full) <- NULL
write.csv(coef_table_full,'glm_abundance_full.csv')


# Extract coefficients with confidence intervals
coef_table_null <- as.data.frame(summary(null)$coefficients)
coef_table_null$Term <- rownames(coef_table_null)
rownames(coef_table_null) <- NULL
write.csv(coef_table_null,'glm_abundance_null.csv')

# try mulnm: many models that vary by one variable have similar AICs
# Berman and Anderson: AIC ~2 indistinguishable etc
AIC(full)

library(MuMIn)
options(na.action = "na.fail")
comparison <- dredge(full)
comparison
write.csv(data.frame(comparison),'glm_dredging.csv')
# 
# ## Abundance estimates from full model at each site and year
# # Load necessary packages
# library(dplyr)
# library(ggplot2)
# 
# # Ensure correct types and levels
# new_data <- count_data %>%
#   dplyr::select(Block, year) %>%
#   distinct() %>%
#   mutate(
#     Block = factor(Block, levels = levels(count_data$Block)),
#     year = factor(year, levels = levels(count_data$year)),
#     slope_scaled = as.numeric(mean(count_data$slope_scaled, na.rm = TRUE)),
#     canopy_scaled = as.numeric(mean(count_data$canopy_scaled, na.rm = TRUE)),
#     ba_scaled = as.numeric(mean(count_data$ba_scaled, na.rm = TRUE))
#   )
# 
# # Predict on log (link) scale with SE
# pred <- predict(full, newdata = new_data, type = "link", se.fit = TRUE)
# 
# # Back-transform to response scale and add CI
# new_data <- new_data %>%
#   mutate(
#     fit = pred$fit,
#     se = pred$se.fit,
#     estimate = exp(fit),
#     lower = exp(fit - 1.96 * se),
#     upper = exp(fit + 1.96 * se)
#   )
# write.csv(select(new_data,c('Block','year','estimate','lower','upper')),'full_abundance_glm_estimates.csv')
# 
# library(ggplot2)
# 
# ggplot(new_data, aes(x = as.numeric(as.character(year)), y = estimate, color = Block)) +
#   geom_line(size = 1.2) +
#   geom_point(size = 2) +
#   geom_ribbon(aes(ymin = lower, ymax = upper, fill = Block), 
#               alpha = 0.2, color = NA) +
#   labs(
#     title = "Estimated Abundance (n_individuals) Over Time by Block",
#     x = "Year",
#     y = "Estimated Abundance",
#     color = "Block",
#     fill = "Block"
#   ) +
#   theme_minimal(base_size = 14)
# ggsave("abundance_by_block_year.pdf", width = 8, height = 6)
# 

## Plot covariate relationships


# Get effect of canopy
pred <- ggpredict(full, terms = "canopy_scaled")

# Unscale x-axis
canopy_mean <- mean(count_data$canopy_cover_50m_avg)
canopy_sd   <- sd(count_data$canopy_cover_50m_avg)
# Get partial effect from model
pred_canopy <- ggpredict(full, terms = "canopy_scaled")

# Back-transform x-axis to original canopy values
pred_canopy$x_unscaled <- pred_canopy$x * canopy_sd + canopy_mean

# Plot
ggplot(pred_canopy, aes(x = x_unscaled, y = predicted)) +
  # Partial effect line and CI ribbon
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), fill = "blue", alpha = 0.2) +
  
  # Add raw data points
  geom_point(data = count_data, 
             aes(x = canopy_cover_50m_avg, y = n_individuals),
             inherit.aes = FALSE, alpha = 0.5, color = "black") +
  
  # Labels and theme
  labs(
    x = "Canopy cover (%)",
    y = "Predicted abundance (n_individuals)",
    title = "Partial Effect of Canopy Cover with Observed Data"
  ) +
  theme_minimal()
