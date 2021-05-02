
library(caret) 
library(tidyverse)
library(future.apply)
library(furrr)

setwd("~/shared_space/sworthin/project_109")

# parallelize
future::plan(multicore)

###################################################################
# repeated holdout

ntrials <- 2e3

# function to iterate over sample size, error, split, and repeats
HO_reps <- function(N, errSD, p, nrep) {
  
  create_data <- function(n, err) {
    set.seed(1)
    d <- data.frame(replicate(5, rnorm(n)))
    d$Y <- with(d, 1 + 2*X1 + 1.3*X2 + 0.9*X3 + 1.5*X4 + 0.6*X5 + rnorm(n, sd = err)) 
    return(d)
  }    
  dat <- create_data(n = N, err = errSD)
  
  rm(.Random.seed, envir = .GlobalEnv) # remove seed from global environment
  
  ctrl_reps <- trainControl(
    method = "LGOCV", 
    p = p,
    number = nrep,
    allowParallel = TRUE,
    seeds = NULL
  )
  
  model <- train(
    form = Y ~ .,
    method = "lm",
    trControl = ctrl_reps,
    data = dat
  )
  
  return(model$results["Rsquared"])   
}

# parameter grid
pgrid <- expand.grid(
  samp_size = c(50, 100, 200, 500, 1000, 5000, 10000, 20000), 
  error = c(1, 2, 3, 6),
  p_num = c(0.5, 0.7, 0.9),  
  rep_num = c(1, 10, 100, 500)  
)

# fit the models over the parameters
parameter_search <- function(grid = pgrid, FUN = HO_reps) {
  future_pmap(
    .l = list(grid$samp_size,
              grid$error,
              grid$p_num,
              grid$rep_num),
    .f = ~ FUN(N = ..1, 
               errSD = ..2, 
               p = ..3, 
               nrep = ..4),
    .options = furrr_options(seed = TRUE),
    .progress = TRUE
  )
}

# replicate function call 2000 times
model_trials <- future_replicate(
  n = ntrials, 
  expr = parameter_search(), 
  simplify = FALSE
) 

# add trial to parameter grid
expand.grid.df <- function(...) Reduce(function(...) merge(..., by=NULL), list(...))
pgrid_trial <- expand.grid.df(
  pgrid,
  data.frame(trial = 1:ntrials)
)

# extract performance metrics (R2)
performance <- model_trials %>% 
  flatten() %>%
  imap(~ mutate(.x, N = pgrid_trial$samp_size[.y],
                error = pgrid_trial$error[.y],
                P = pgrid_trial$p_num[.y], 
                reps = pgrid_trial$rep_num[.y],
                trial = pgrid_trial$trial[.y])) %>%
  bind_rows() 

###################################################################
# graphs for repeated holdout

performance <- performance %>%
  mutate(N_fac = factor(N, levels = c(50, 100, 200, 500, 1000, 5000, 10000, 20000)),
         error_fac = factor(error, levels = c(1, 2, 3, 6), 
                            labels = c("1 SD", "2 SD", "3 SD", "6 SD")),
         P_fac = factor(P, levels = c(0.5, 0.7, 0.9), 
                        labels = c("Split = 0.5", "Split = 0.7", "Split = 0.9")),
         reps_fac = factor(reps, levels = c(1, 10, 100, 500), 
                           labels = c("Reps = 1", "Reps = 10", "Reps = 100", "Reps = 500")))

plot1 <- ggplot(performance, aes(x = N_fac, y = Rsquared, color = error_fac)) +
  geom_boxplot(outlier.size = 0.1) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_brewer(palette = "Set1", name = "Error") +
  facet_grid(reps_fac ~ P_fac) +
  labs(x = "Sample size", y = bquote(R^2)) +
  theme_classic() +
  theme(legend.position = "top",
        legend.box.margin = margin(0, 0, -10, 0),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        strip.background = element_rect(fill = "grey85", color = NA))
ggsave(plot1, file = "HO_boxplots.pdf", height = 6, width = 6)

# stability as range of central 95 percentiles
performance_agg <- performance %>%
  group_by(N, N_fac, error_fac, P_fac, reps, reps_fac) %>%
  summarize(stability = diff(quantile(Rsquared, probs = c(0.025, 0.975)))) %>%
  mutate(reps_fac = factor(reps_fac, 
                           levels = c("Reps = 1", "Reps = 10", "Reps = 100", "Reps = 500"), 
                           labels = c("1", "10", "100", "500")))

# x-axis common log scale for data and linear scale for labels    
plot2 <- ggplot(performance_agg, aes(x = N, y = stability, color = reps_fac)) +
  geom_line() +
  scale_x_log10(breaks = c(50, 100, 200, 500, 1000, 5000, 10000, 20000)) +
  scale_color_brewer(palette = "Set1", name = "Repetitions") +
  facet_grid(error_fac ~ P_fac, scales = "free_y", space = "free_y") + 
  labs(x = "Sample size (ln)", y = bquote("Stability of"~R^2)) +
  theme_classic() +
  theme(legend.position = "top",
        legend.box.margin = margin(0, 0, -10, 0),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        strip.background = element_rect(fill = "grey85", color = NA))
ggsave(plot2, file = "HO_stability.pdf", height = 6.5, width = 6.5) 
