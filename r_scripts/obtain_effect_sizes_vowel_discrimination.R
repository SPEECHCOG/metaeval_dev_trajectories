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

calculate_mean_effect <- function(effect_sizes_per_contrast, effect){
  mean_es = mean(effect_sizes_per_contrast[[effect]])
  sd_es = sd(effect_sizes_per_contrast[[effect]])
  return(list('mean_es'=mean_es, 'sd_es'=sd_es))
}

# Calculate effect sizes for Native and Non-native contrasts
es = 'g'

obtain_effects_for_all_epochs <- function(folder, model, contrasts_type, es){
  es_epochs <- list()
  for (epoch in 0:10) {
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
    es_epoch <- calculate_mean_effect(es_all_contrasts, es)
    
    es_epochs <- append(es_epochs, list(es_epoch))
  }
  return (es_epochs)
}


create_dev_trajectories_plot <- function(effects_lists, title){
  ds <- c()
  sd <- c()
  for (epoch in 1:11){
    ds <- c(ds, effects_lists[[epoch]]$mean_es)
    sd <- c(sd, effects_lists[[epoch]]$sd_es)
  }
  print(ds)
  print(sd)
  
  plotting_data <- data.frame(
    hours = c(0:10),
    d = ds,
    sd = sd
  )
  print(plotting_data)
  
  p=ggplot(plotting_data, aes(y=d, x=hours)) +
    #Add data points and color them black
    geom_point(size=3, shape=16) +
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

# Development of ES (0-100 h training)
get_es_change_plot <- function(es_dev, lb_y=NA, ub_y=NA, pos_legend=NA){
  pos_label_x <- es_dev$pos_label_x
  pos_label_y <- es_dev$pos_label_y
  p=ggplot(es_dev, aes(y=es, x=hours, group=model, colour=model, 
                       linetype=model, label=significance)) +
    #Add data points and color them black
    geom_point(colour = 'black', size=3, shape=16) +
    geom_text(size=10, nudge_x = pos_label_x, nudge_y = pos_label_y, show.legend = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    #Give y-axis a meaningful label
    xlab('\nInput duration (hours)') +
    ylab('Effect size\n') +
    geom_smooth(se=FALSE, method=lm)
  
  if(!is.na(lb_y) & !is.na(ub_y)){
    p <- p + scale_y_continuous(expand = c(0, 0), limits = c(lb_y, ub_y),
                                breaks = seq(lb_y, ub_y, 1)) +
      coord_cartesian(clip = "off")
  }
  if (is.na(pos_legend)){
    pos_legend = c(0.8, 0.2)
  }
  p + theme(legend.position = pos_legend, text = element_text(size=18), 
            axis.line = element_line(color='black', size=1)) +
    labs(colour='Model:', linetype='Model:', label="")
}

# Native contrasts
es_dev_nat <- data.frame(hours = c('0', '100', '0', '100'),
                     es = c(apc_untrained_es_nat$mean_es, apc_es_nat$mean_es, cpc_untrained_es_nat$mean_es, cpc_es_nat$mean_es),
                     significance = factor(c('s.', 's.', 's.', 's.'), labels=labels_significance, levels=c('n.s.', 's.')),
                     pos_label_x = c(-0.1,0.05,-0.1, 0.05),
                     pos_label_y = c(-0.1,0.02,0.01, 0.02),
                     model = c('APC', 'APC', 'CPC', 'CPC'))

get_es_change_plot(es_dev_nat, -1.5,2.5)

# Non-native contrasts
es_dev_nonnat <- data.frame(hours = c('0', '100', '0', '100'),
                         es = c(apc_untrained_es_nonnat$mean_es, apc_es_nonnat$mean_es, cpc_untrained_es_nonnat$mean_es, cpc_es_nonnat$mean_es),
                         significance = factor(c('s.', 's.', 's.', 's.'), labels=labels_significance, levels=c('n.s.', 's.')),
                         pos_label_x = c(-0.1,0.08,-0.1, 0.08),
                         pos_label_y = c(0.01,0.02,0.01, 0.02),
                         model = c('APC', 'APC', 'CPC', 'CPC'))

get_es_change_plot(es_dev_nonnat, -1, 3.2, c(0.2, 0.8))

