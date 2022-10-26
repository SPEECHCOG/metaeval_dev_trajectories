# IDS preference test
# Calculation of effect size based on attentional preference scores 
# from APC and CPC models. 

library(tidyverse)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

get_trial_response <-function(measurements){
  trial_scores <- measurements %>%
    group_by(file_name, trial_type) %>%
    summarise(alpha = mean(attentional_preference_score, na.rm = TRUE)) 
  return(trial_scores)
}

calculate_means <- function(score_file_path){
  # one effect size per paired trial
  measurements <- read.csv(score_file_path, sep = ';')
  
  trial_scores <- get_trial_response(measurements)
  
  cond_statistics <- trial_scores %>% 
    group_by(trial_type) %>%
    summarise(mean = mean(alpha),
              sd = sd(alpha))
  ads = cond_statistics[cond_statistics$trial_type=='ADS',]
  ids = cond_statistics[cond_statistics$trial_type=='IDS',]
  
  return(list('ads'=ads$mean, 'ids'=ids$mean))
}

obtain_effects_for_all_epochs <- function(folder, model){
  es_epochs <- list()
  steps = c(0, 562, 1125, 1688, 2251, 2814, 3377, 3940, 4503, 5066, 5629)
  steps = c(steps, 1:10)
  for (epoch in steps) {
    results_path = paste(folder, model, '/', as.character(epoch), '/ids/attentional_preference_scores.csv', sep='')
    es_epoch = calculate_means(results_path)
    es_epochs <- append(es_epochs, list(es_epoch))
  }
  return (es_epochs)
}

create_trajectories_plot <- function(means_list, title){
  ads <- c()
  ids <- c()
  total_steps <- length(means_list)
  for (epoch in 1:total_steps){
    ads <- c(ads, means_list[[epoch]]$ads)
    ids <- c(ids, means_list[[epoch]]$ids)
  }
  print(ads)
  print(ids)
  
  time_steps <- c(0:10)/10
  time_steps <- c(time_steps, 1:10)
  
  print(total_steps)
  print(length(time_steps))
  
  plotting_data <- data.frame(
    hours = time_steps,
    means = c(ads, ids),
    type = c(rep(c('ads'), each=length(ads)), rep(c('ids'), each=length(ids)))
  )
  print(plotting_data)
  
  p=ggplot(plotting_data, aes(y=means, x=hours, group=type, colour=type)) +
    #Add data points and color them black
    geom_point(size=1, shape=16) +
    # geom_text(size=10, nudge_x = pos_label_x, nudge_y = pos_label_y, show.legend = FALSE) +
    # coord_cartesian(clip = "off") +
    # geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    #Give y-axis a meaningful label
    xlab('\nEpochs') +
    ylab('Loss\n') +
    geom_smooth(se=FALSE, method=lm)+
    ggtitle(title) + 
    theme(axis.line = element_line(color='black', size=1), plot.title = element_text(hjust = 0.5))
  p
}

apc_means <- obtain_effects_for_all_epochs('tests_results/', 'apc')
cpc_means <- obtain_effects_for_all_epochs('tests_results/', 'cpc')

create_trajectories_plot(apc_means, 'APC')
create_trajectories_plot(cpc_means, 'CPC')

#------------------------------------------------------------------------------#
calculate_means_verification <- function(score_file_path){
  # one effect size per paired trial
  measurements <- read.csv(score_file_path, sep = ';')
  
  trial_scores <- get_trial_response(measurements)
  mean = mean(trial_scores$alpha)
  sd = sd(trial_scores$alpha)
  return(mean)
}

obtain_effects_verification <- function(folder, model){
  es_epochs <- c()
  steps = c(0, 562, 1125, 1688, 2251, 2814, 3377, 3940, 4503, 5066, 5629)
  steps = c(steps, 1:10)
  for (epoch in steps) {
    results_path = paste(folder, model, '/', as.character(epoch), '/ids/attentional_preference_scores.csv', sep='')
    es_epoch = calculate_means_verification(results_path)
    es_epochs <- c(es_epochs, es_epoch)
  }
  return (es_epochs)
}

create_trajectories_plot_verification <- function(means_list, title){

  total_steps <- length(means_list)

  time_steps <- c(0:10)/10
  time_steps <- c(time_steps, 1:10)
  
  print(total_steps)
  print(length(time_steps))
  
  plotting_data <- data.frame(
    hours = time_steps,
    means = means_list
  )
  
  print(plotting_data)
  
  p=ggplot(plotting_data, aes(y=means, x=hours)) +
    #Add data points and color them black
    geom_point(size=1, shape=16) +
    # geom_text(size=10, nudge_x = pos_label_x, nudge_y = pos_label_y, show.legend = FALSE) +
    # coord_cartesian(clip = "off") +
    # geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    #Give y-axis a meaningful label
    xlab('\nEpochs') +
    ylab('Loss\n') +
    geom_smooth(se=FALSE, method=lm)+
    ggtitle(title) + 
    theme(axis.line = element_line(color='black', size=1), plot.title = element_text(hjust = 0.5))
  p
}

apc_means_ver <- obtain_effects_verification('tests_results_val/', 'apc')
cpc_means_ver <- obtain_effects_verification('tests_results_val/', 'cpc')

create_trajectories_plot_verification(apc_means_ver, 'APC')
create_trajectories_plot_verification(cpc_means_ver, 'CPC')
