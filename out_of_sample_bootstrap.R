
library(tidyverse)
library(future.apply)
library(furrr)

setwd("~/shared_space/sworthin/project_109")

# parallelize
future::plan(multicore)

###################################################################
# out-of-sample bootstrap

ntrials <- 2e3

# function to iterate over sample size, error, and iterations
boot_oos <- function(N, errSD, iter) {
  
  create_data <- function(n, err) {
    set.seed(1)
    d <- data.frame(replicate(5, rnorm(n)))
    d$Y <- with(d, 1 + 2*X1 + 1.3*X2 + 0.9*X3 + 1.5*X4 + 0.6*X5 + rnorm(n, sd = err)) 
    return(d)
  }    
  dat <- create_data(n = N, err = errSD)
  
  rm(.Random.seed, envir = .GlobalEnv) # remove seed from global environment
  
  # container	
  Rsquared <- vector(mode = "numeric", length = iter)
  
  for(i in 1:iter) {
    
    # generate a bootstrap sample with replacement
    indices <- sample(nrow(dat), replace = TRUE)
    
    # generate training dataset using the bootstrap sample
    training <- dat[indices, ]
    
    # generate testing dataset of instances not included in bootstrap sample
    testing <- dat[-unique(indices), ]
    
    # fit linear model
    model <- lm(Y ~ ., data = training)
    
    # calculate out-of-sample predictions using the testing dataset
    Ynew <- predict(model, newdata = testing)
    
    # performance
    Rsquared[i] <- 1 - (sum((testing$Y - Ynew)^2) / sum((testing$Y - mean(testing$Y))^2))
    
  }
  
  return(mean(Rsquared))   
}

# parameter grid
pgrid <- expand.grid(
  samp_size = c(50, 100, 200, 500, 1000), 
  error = c(1, 2, 3, 6),
  iter_num = c(25, 100, 200, 500, 1000)   
)

# fit the models over the parameters
parameter_search <- function(grid = pgrid, FUN = boot_oos) {
  future_pmap(
    .l = list(grid$samp_size,
              grid$error,
              grid$iter_num),
    .f = ~ FUN(N = ..1, 
               errSD = ..2, 
               iter = ..3),
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
                iter = pgrid_trial$iter_num[.y], 
                trial = pgrid_trial$trial[.y])) %>%
  bind_rows() 

###################################################################
# graphs for out-of-sample bootstrap

library(gridExtra)
library(ggpubr)
library(cowplot)

performance <- performance %>%
  mutate(N_fac = factor(N, levels = c(50, 100, 200, 500, 1000)),
         error_fac = factor(error, 
                            levels = c(1, 2, 3, 6), 
                            labels = c("1 SD", "2 SD", "3 SD", "6 SD")),
         iter_fac = factor(iter, 
                           levels = c(25, 100, 200, 500, 1000), 
                           labels = c("Iter = 25", "Iter = 100", "Iter = 200", "Iter = 500", "Iter = 1000"))
  )

plot1 <- ggplot(performance, aes(x = N_fac, y = Rsquared, color = error_fac)) +
  geom_boxplot(outlier.size = 0.1) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_brewer(palette = "Set1", name = "Error") +
  facet_wrap( ~ iter_fac, nrow = 1) +
  labs(x = "Sample size", y = bquote(R^2)) +
  theme_classic() +
  theme(legend.position = "top",
        legend.box.margin = margin(0, 0, -10, 0),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        strip.background = element_rect(fill = "grey85", color = NA))

# stability as range of central 95 percentiles
performance_agg <- performance %>%
  group_by(N, N_fac, error_fac, iter_fac) %>%
  summarize(stability = diff(quantile(Rsquared, probs = c(0.025, 0.975)))) %>%
  mutate(iter_fac = factor(iter_fac, 
                           levels = c("Iter = 25", "Iter = 100", "Iter = 200", "Iter = 500", "Iter = 1000"), 
                           labels = c(25, 100, 200, 500, 1000)))

# x-axis common log scale for data and linear scale for labels    
plot2 <- ggplot(performance_agg, aes(x = N, y = stability, color = iter_fac)) +
  geom_line() +
  scale_x_log10(breaks = c(50, 100, 200, 500, 1000)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.05)) +
  scale_color_brewer(palette = "Set1", name = "Iterations") +
  facet_wrap(~ error_fac, nrow = 1) +
  labs(x = "Sample size", y = bquote("Stability of"~R^2)) +
  theme_classic() +
  theme(legend.position = "top",
        legend.box.margin = margin(0, 0, -10, 0),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        strip.background = element_rect(fill = "grey85", color = NA))

# arrange plots using arrangeGrob
# returns a gtable (gt)
gt <- arrangeGrob(plot1, plot2, ncol = 1, nrow = 2)

# add labels to the arranged plots
p <- as_ggplot(gt) + 
  draw_plot_label(
    label = c("A", "B"), size = 12,
    x = c(0, 0), y = c(1, 0.5), # add labels
    fontface = "plain") 

ggsave(p, file = "Boot_oos_combined.pdf", height = 6, width = 7) 
