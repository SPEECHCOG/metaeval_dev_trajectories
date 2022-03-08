# It plots the developmental trajectories of models for the two linguistic 
# capabilities (IDS preference and Vowel discrimination)
library(extrafont)
library(tidyverse)
library(ggplot2)

options(dplyr.summarise.inform = FALSE)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source('obtain_effect_sizes_ids.R')
source('obtain_effect_sizes_vowel_discrimination.R')
source('visualise_infants_trajectories.R')

create_plot_trajectories <- function(effects_df, title){
  
  p=ggplot(effects_df, aes(y=d, x=days, group=capability, colour=capability,
                           shape=checkpoint)) +
    geom_point(size=1.5) +
    guides("shape"="none") +
    scale_shape_manual(values =c("batch"=1, "epoch"=20), )+
    scale_color_manual(values = c("IDS preference" = "#00BA38",
                                  "Native discrimination" = "#F8766D",
                                  "Non-native discrimination" = "#619CFF")) +
    scale_y_continuous(expand = c(0, 0), limits = c(-1, 4.5),
                       breaks = seq(-1, 4.5, 0.5)) +
    coord_cartesian(clip = "off") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    xlab('\nSimulated Days') +
    ylab('Effect size\n') +
    geom_smooth(se=FALSE, method="lm")+
    ggtitle(title) + 
    theme(legend.position = c(0.7,0.85), legend.text=element_text(family="LM Roman 10", size=14),
          legend.title=element_blank(), text = element_text(family="LM Roman 10", size=18), 
          axis.line = element_line(color='black', size=1), 
          plot.title = element_text(hjust = 0.5)) 
  return(p)
}

get_overlapped_plot <- function(model_effects, title){
  overlapped_plot <- infants_plot +
    geom_point(size=1.5, data=model_effects, aes(y=d, x=days, group=capability,
                                                 colour=capability, 
                                                 shape=checkpoint)) +
    guides("shape"="none") +
    scale_shape_manual(values =c("batch"=1, "epoch"=20), )+
    geom_smooth(size=0.9, se=FALSE, method="lm", data=model_effects, 
                aes(y=d, x=days, group=capability, colour=capability)) + 
    xlab('\nSimulated Days') +
    ggtitle(title)
  return(overlapped_plot)
}


alpha = 0.05
folder = "tests_results/"
es = "g"  # Effect size calculation for vowel discrimination test
models = c("apc", "cpc")
apc_results = data.frame()
cpc_results = data.frame()
for (model in models){
  ids_df <- get_ids_effects_dataframe(folder, model, alpha)
  ids_df$capability = "IDS preference"
  
  # native
  contrasts_type = "native"
  
  vowel_df <- get_vowel_disc_effects_dataframe(folder, model, contrasts_type, es, alpha)
  vowel_df$capability = "Native discrimination"
  
  # non-native
  contrasts_type = "non_native"
  vowel2_df <- get_vowel_disc_effects_dataframe(folder, model, contrasts_type, es, alpha)
  vowel2_df$capability = "Non-native discrimination"
  
  all_capabilities = rbind(ids_df, vowel_df, vowel2_df)
  if(model == 'apc'){
    apc_results <- all_capabilities
  }else{
    cpc_results <- all_capabilities
  }
  
  plot_traj <- create_plot_trajectories(all_capabilities, toupper(model))
  print(plot_traj)
  overlapped_plot_traj <- get_overlapped_plot(all_capabilities, toupper(model))
  print(overlapped_plot_traj)
}

