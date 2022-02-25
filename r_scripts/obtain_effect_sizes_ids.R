# IDS preference test
# Calculation of effect size based on attentional preference scores 
# from APC and CPC models. 

library(tidyverse)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

get_ids_trial_response <-function(measurements){
  trial_scores <- measurements %>%
    group_by(file_name, trial_type) %>%
    summarise(alpha = mean(attentional_preference_score, na.rm = TRUE)) 
  return(trial_scores)
}

get_ids_t_statistics <- function(trial_scores){
  t_test <- t.test(trial_scores[trial_scores$trial_type=='IDS',]$alpha, 
                   trial_scores[trial_scores$trial_type=='ADS',]$alpha)
  return(t_test)
}

get_ids_d_var <- function(d, trial_scores){
  ids_n <- nrow(trial_scores[trial_scores$trial_type=='IDS',])
  ads_n <- nrow(trial_scores[trial_scores$trial_type=='ADS',])
  
  d_var <- (ids_n+ads_n)/(ids_n*ads_n)+(d^2)/(2*(ids_n+ads_n))
  return(d_var)
}

calculate_ids_standardised_mean_gain <- function(score_file_path){
  # one effect size per paired trial
  measurements <- read.csv(score_file_path, sep = ';')
  
  trial_scores <- get_ids_trial_response(measurements)
  
  cond_statistics <- trial_scores %>% 
    group_by(trial_type) %>%
    summarise(mean = mean(alpha),
              sd = sd(alpha),
              n = n())
  ads = cond_statistics[cond_statistics$trial_type=='ADS',]
  ids = cond_statistics[cond_statistics$trial_type=='IDS',]
  
  d = (ids$mean - ads$mean) / sqrt((ids$sd^2 + ads$sd^2)/2)
  g = d * (1 - (3/(4*(ads$n + ids$n) - 9)))
  
  t_test <- get_ids_t_statistics(trial_scores)
  d_var <- get_ids_d_var(d, trial_scores)
  
  # 95% CI
  ci_lb <- d - 1.959964*sqrt(d_var)
  ci_ub <- d + 1.959964*sqrt(d_var)
  
  return(list('d'=d, 't'= t_test$statistic, 'p-value'=t_test$p.value, 
              'ci_lb'=ci_lb, 'ci_ub'=ci_ub, 'g'=g))
}

obtain_ids_effects_for_all_epochs <- function(folder, model){
  es_epochs <- list()
  steps = c(0, 562, 1125, 1688, 2251, 2814, 3377, 3940, 4503, 5066, 5629)
  steps = c(steps, 1:10)
  for (epoch in steps) {
    results_path = paste(folder, model, '/', as.character(epoch), '/ids/attentional_preference_scores.csv', sep='')
    es_epoch = calculate_ids_standardised_mean_gain(results_path)
    es_epochs <- append(es_epochs, list(es_epoch))
  }
  return (es_epochs)
}

get_ids_effects_dataframe <- function(folder, model, alpha){
  effects_list <- obtain_ids_effects_for_all_epochs(folder, model)
  ds <- c()
  p_values <- c()
  total_steps <- length(effects_list)
  for (epoch in 1:total_steps){
    ds <- c(ds, effects_list[[epoch]]$g)
    p_values <- c(p_values, effects_list[[epoch]]$'p-value')
  }
  
  days <- c(0:10)*1.73  # days represented by 10 hours of speech
  days <- c(days, c(1:9)*17.3, 9*17.3 + 10.3) # total days represented by 960 hours of speech. last chunk only contains 60 hours of speech
  
  df <- data.frame(
    days = days,
    d = ds,
    significant = p_values <= alpha
  )
  return (df)
}

create_dev_trajectories_plot <- function(effects_lists, alpha, title){
  ds <- c()
  p_values <- c()
  total_steps <- length(effects_lists)
  for (epoch in 1:total_steps){
    ds <- c(ds, effects_lists[[epoch]]$d)
    p_values <- c(p_values, effects_lists[[epoch]]$'p-value')
  }
  
  days <- c(0:10)*1.73  # days represented by 10 hours of speech
  days <- c(days, c(1:9)*17.3, 9*17.3 + 10.3) # total days represented by 960 hours of speech. last chunk only contains 60 hours of speech
  
  plotting_data <- data.frame(
    days = days,
    d = ds,
    significant = p_values <= alpha
  )
  
  plotting_data <- plotting_data %>% filter(days>0)
  
  p=ggplot(plotting_data, aes(y=d, x=days)) +
    #Add data points and color them black
    geom_point(size=2) +
    # geom_text(size=10, nudge_x = pos_label_x, nudge_y = pos_label_y, show.legend = FALSE) +
    #scale_y_continuous(expand = c(0, 0), limits = c(-1, 2.3),
    #                   breaks = seq(-1, 2.3, 1)) +
    coord_cartesian(clip = "off") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    #Give y-axis a meaningful label
    xlab('\nInput duration (simulated days)') +
    ylab('Effect size\n') +
    geom_smooth(se=FALSE, method=lm)+
    ggtitle(title) + 
    theme(axis.line = element_line(color='black', size=1), plot.title = element_text(hjust = 0.5))
  p
  
}


# effect sizes plots

# ES comparison with infants' ES
# IDS meta-analysis: 0.43 [0.33-0.53]


