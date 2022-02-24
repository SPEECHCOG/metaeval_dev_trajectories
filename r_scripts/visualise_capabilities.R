# It plots the developmental trajectories of models for the two linguistic 
# capabilities (IDS preference and Vowel discrimination)

library(tidyverse)
library(ggplot2)
options(dplyr.summarise.inform = FALSE)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source('obtain_effect_sizes_ids.R')
source('obtain_effect_sizes_vowel_discrimination.R')

create_plot_trajectories <- function(effects_df, title){
  
  p=ggplot(effects_df, aes(y=d, x=days, group=capability, colour=capability, shape=significant)) +
    geom_point(size=2) +
    scale_color_manual(values = c("IDS preference" = "#F8766D",
                                  "Native discrimination" = "#619CFF",
                                  "Non-native discrimination" = "#00BA38")) +
    scale_y_continuous(expand = c(0, 0), limits = c(-1, 3),
                       breaks = seq(-1, 3, 0.5)) +
    coord_cartesian(clip = "off") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    xlab('\nInput duration (simulated days)') +
    ylab('Effect size\n') +
    geom_smooth(se=FALSE, method=lm)+
    ggtitle(title) + 
    theme(axis.line = element_line(color='black', size=1), plot.title = element_text(hjust = 0.5))
  return(p)
}

alpha = 0.05
folder = "tests_results/"
es = "g"  # Effect size calculation for vowel discrimination test
models = c("apc", "cpc")
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
  
  plot <- create_plot_trajectories(all_capabilities, toupper(model))
  print(plot)
}



