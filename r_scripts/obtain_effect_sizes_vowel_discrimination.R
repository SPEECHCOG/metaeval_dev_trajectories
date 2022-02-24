# Vowel discrimination test
# Calculation of effect size based on DTW distances (using encoded stimuli)
# from APC and CPC models. 

library(tidyverse)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

get_condition_mean <- function(measurements){
  mean_per_condition <- measurements %>% 
    group_by(contrast, language, condition) %>%
    summarise(mean = mean(distance, na.rm = TRUE),
              sd = sd(distance, na.rm = TRUE),
              n = n())
  return(mean_per_condition)
}

get_t_statistics <- function(contrast_measurements){
  tryCatch(
    expr = {
      t_test <- t.test(contrast_measurements[contrast_measurements$condition=='different',]$distance,
                       contrast_measurements[contrast_measurements$condition=='same',]$distance)
      return(t_test)
    },
    error = function(e){
      t_test <- data.frame(statistic=c(NA), p.value=c(NA))
      return(t_test)
    }
  )
}

calculate_standardised_mean_gain_per_contrast <- function(dtw_distances_file_path){
  # one effect size per paired trial
  measurements <- read.csv(dtw_distances_file_path, sep = ';')
  
  condition_statistics <- get_condition_mean(measurements)
  # print(condition_statistics)
  
  total_contrasts <- condition_statistics  %>% group_by(contrast, language) %>% 
    select(contrast, language) %>% filter(row_number()==1)
  
  effect_sizes_contrasts <- condition_statistics %>% 
    group_by(contrast, language) %>% 
    summarise(d=(mean[condition=='different']-mean[condition=='same'])/
                sqrt((sd[condition=='different']^2+sd[condition=='same']^2)/2),
              n1 = n[condition=='different'], n2=n[condition=='same'])
  effect_sizes_contrasts <- effect_sizes_contrasts %>% 
    mutate(g = d * (1 - 3/(4*(n1+n2-2) - 1)))
  effect_sizes_contrasts$t <- 0
  effect_sizes_contrasts$p_value <- 0
  
  for(i in 1:nrow(total_contrasts)){
    contrast_tmp = total_contrasts[i, ]$contrast
    language_tmp = total_contrasts[i, ]$language
    
    contrast_measurements <- measurements %>% filter(contrast == contrast_tmp, language == language_tmp)
    t_test_contrast <- get_t_statistics(contrast_measurements)
    # print(t_test_contrast)
    effect_sizes_contrasts[effect_sizes_contrasts$contrast==contrast_tmp & effect_sizes_contrasts$language==language_tmp,]$t = t_test_contrast$statistic
    effect_sizes_contrasts[effect_sizes_contrasts$contrast==contrast_tmp & effect_sizes_contrasts$language==language_tmp,]$p_value = t_test_contrast$p.value
  }
  
  return(effect_sizes_contrasts)
}

calculate_mean_effect <- function(effect_sizes_per_contrast, effect, alpha){
  mean_es = mean(effect_sizes_per_contrast[[effect]])
  sd_es = sd(effect_sizes_per_contrast[[effect]])
  signficant = !any(effect_sizes_per_contrast[["p_value"]]>alpha)
  return(list('mean_es'=mean_es, 'sd_es'=sd_es, 'significant'=signficant))
}

# Calculate effect sizes for Native and Non-native contrasts
es = 'g'

obtain_vowel_effects_for_all_epochs <- function(folder, model, contrasts_type, es, alpha){
  es_epochs <- list()
  steps = c(0, 562, 1125, 1688, 2251, 2814, 3377, 3940, 4503, 5066, 5629)
  steps = c(steps, 1:10)
  for (epoch in steps) {
    if (contrasts_type=='native') {
      results_path_c = paste(folder, model, '/', as.character(epoch), '/vowel_disc/dtw_distances_hc_native.csv', sep='')
      results_path_ivc = paste(folder, model, '/', as.character(epoch), '/vowel_disc/dtw_distances_ivc_native.csv', sep='')
    }else{
      results_path_c = paste(folder, model, '/', as.character(epoch), '/vowel_disc/dtw_distances_oc_non_native.csv', sep='')
      results_path_ivc = paste(folder, model, '/', as.character(epoch), '/vowel_disc/dtw_distances_ivc_non_native.csv', sep='')
    }
    
    es_contrasts_c <- calculate_standardised_mean_gain_per_contrast(results_path_c)
    es_contrasts_ivc <- calculate_standardised_mean_gain_per_contrast(results_path_ivc)
    es_all_contrasts <- rbind(es_contrasts_c, es_contrasts_ivc)
    es_epoch <- calculate_mean_effect(es_all_contrasts, es, alpha)
    
    es_epochs <- append(es_epochs, list(es_epoch))
  }
  return (es_epochs)
}

get_vowel_disc_effects_dataframe <- function(folder, model, contrasts_type, es, alpha){
  effects_list <- obtain_vowel_effects_for_all_epochs(folder, model, contrasts_type, es, alpha)
  ds <- c()
  significant <- c()
  total_steps <- length(effects_list)
  for (epoch in 1:total_steps){
    ds <- c(ds, effects_list[[epoch]]$mean_es)
    significant <- c(significant, effects_list[[epoch]]$significant)
  }
  
  days <- c(0:10)*1.73  # days represented by 10 hours of speech
  days <- c(days, c(1:9)*17.3, 9*17.3 + 10.3) # total days represented by 960 hours of speech. last chunk only contains 60 hours of speech
  
  df <- data.frame(
    days = days,
    d = ds,
    significant = significant
  )
  return (df)
}

create_dev_trajectories_plot <- function(effects_lists, title){
  ds <- c()
  sd <- c()
  non_significant <- c()
  for (epoch in 1:11){
    ds <- c(ds, effects_lists[[epoch]]$mean_es)
    sd <- c(sd, effects_lists[[epoch]]$sd_es)
    non_significant <- c(non_significant, effects_lists[[epoch]]$"non_significant")
  }
  print(ds)
  print(sd)
  
  plotting_data <- data.frame(
    hours = c(0:10),
    d = ds,
    sd = sd,
    non_significant = non_significant
  )
  print(plotting_data)
  
  p=ggplot(plotting_data, aes(y=d, x=hours)) +
    #Add data points and color them black
    geom_point(size=3, shape=non_significant) +
    # geom_errorbar(aes(x=hours, ymin=d-sd, ymax=d+sd), width=0.4, colour="gray", alpha=0.8) +
    # geom_text(size=10, nudge_x = pos_label_x, nudge_y = pos_label_y, show.legend = FALSE) +
    scale_y_continuous(expand = c(0, 0), limits = c(-1, 4),
                       breaks = seq(-1, 2.3, 1)) +
    coord_cartesian(clip = "off") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    #Give y-axis a meaningful label
    xlab('\nInput duration (hours)') +
    ylab('Effect size\n') +
    geom_smooth(se=FALSE, method=lm)+
    ggtitle(title) +
    theme(axis.line = element_line(color='black', size=1), plot.title = element_text(hjust = 0.5))
  p
  
}


# TODO Update Infants' ES! 
# native: 0.42 [0.33-0.51]
# non-native: 0.46 [0.21-0.72]



