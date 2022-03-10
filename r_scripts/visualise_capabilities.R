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

get_overlapped_plot_per_cap <- function(models_effects, cap){
  infants_data = all_capabilities_infants %>% filter(capability==cap)
  models_data = models_effects %>% filter(capability==cap)

  overlapped_plot <- ggplot(infants_data, aes(x = days, y = ds)) + 
    #scale_color_manual(values = c("IDS preference" = "#00BA38",
    #                              "Native discrimination" = "#F8766D",
    #                              "Non-native discrimination" = "#619CFF")) +
    #scale_fill_manual(values = c("IDS preference" = "#00BA38",
    #                             "Native discrimination" = "#F8766D",
    #                             "Non-native discrimination" = "#619CFF")) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    
    scale_size_continuous(guide = "none") +
    scale_y_continuous(expand = c(0, 0), limits = c(-1, 4.5),
                       breaks = seq(-1, 4.5, 0.5)) +
    coord_cartesian(clip = "off") +
    xlab("\nMean age (days)") +
    ylab("Effect Size\n") + 
    
    geom_point(size=1.5, data=models_data, aes(y=d, x=days, 
                                               colour=model)) +
    guides("shape"="none") +
    #scale_shape_manual(values =c("batch"=1, "epoch"=20), )+
    geom_smooth(size=0.9, se=FALSE, method="lm", data=models_data, 
                aes(y=predicted, x=days, colour=model)) + 
    
    ggtitle(cap) + 
    
    theme(legend.position = c(0.7,0.85), 
          legend.text=element_text(family="LM Roman 10", size=14),
          legend.title=element_blank(), 
          text = element_text(family="LM Roman 10", size=18), 
          axis.line = element_line(color='black', size=1), 
          plot.title = element_text(hjust = 0.5))
  
  if(cap=="IDS preference"){
    # IDS 
    overlapped_plot <- overlapped_plot + geom_smooth(method = "lm", 
                                                     size=0.9, se=TRUE, 
                                                     xseq=130:166, 
                                                     colour="#00BA38",
                                                     fill="#00BA38",
                                                     linetype="dotted")
  }else{
    # Vowel discrimination
    overlapped_plot <- overlapped_plot + 
      annotate('ribbon', x = c(1, 166), ymin = mean_es_nat_ci.lb, 
               ymax = mean_es_nat_ci.ub, 
               alpha = 0.5, fill=alpha("#F8766D", alpha=0.2)) +
      annotate('ribbon', x = c(1, 166), ymin = mean_es_nonnat_ci.lb, 
               ymax = mean_es_nonnat_ci.ub, 
               alpha = 0.5, fill=alpha("#619CFF", alpha=0.8)) + 
      geom_line(size=0.9, linetype="dotted", aes(x = days, y = ds,
                                                 colour=capability), data=infants_data) +
      scale_color_manual(values = c("Native discrimination" = "#F8766D",
                                    "Non-native discrimination" = "#619CFF",
                                    
                                    "APC" = "#C4A0D9",
                                    "NCPC" = "#FFAF49",
                                    "Non-native discrimination.APC" = "#7629A0",
                                    "Non-native discrimination.CPC" = "#E27000"))
  }
  return(overlapped_plot)
}

# Calculate effect sizes per model and capability
alpha = 0.05
folder = "tests_results/"
es = "g"  # Effect size calculation for vowel discrimination test
models = c("apc", "cpc")
apc_results = data.frame()
cpc_results = data.frame()
for (model in models){
  ids_models_df <- get_ids_effects_dataframe(folder, model, alpha)
  ids_models_df$capability = "IDS preference"
  
  # native
  contrasts_type = "native"
  
  vowel_df <- get_vowel_disc_effects_dataframe(folder, model, contrasts_type, es, alpha)
  vowel_df$capability = "Native discrimination"
  
  # non-native
  contrasts_type = "non_native"
  vowel2_df <- get_vowel_disc_effects_dataframe(folder, model, contrasts_type, es, alpha)
  vowel2_df$capability = "Non-native discrimination"
  
  all_capabilities_models = rbind(ids_models_df, vowel_df, vowel2_df)
  if(model == 'apc'){
    apc_results <- all_capabilities_models
  }else{
    cpc_results <- all_capabilities_models
  }
}

# Fit linear models
apc_results$predicted = 0
cpc_results$predicted = 0

for(model in models){
  for(capability_name in c("IDS preference", "Native discrimination", "Non-native discrimination")){
    print(capability_name)
    if(model=="apc"){
      subset = apc_results %>% filter(capability==capability_name)
    }else{
      subset = cpc_results %>% filter(capability==capability_name)
    }
    print(paste(toupper(model), ": ", capability_name, sep=""))
    lm_fit = lm(d~days, subset)
    print(summary(lm_fit))
    age_significance <- summary(lm_fit)$coefficients[2,4]
    mean_es <- summary(lm_fit)$coefficients[1,1]
    trajectory <- predict(lm_fit, subset)
    final_prediction = 0
    if(age_significance>alpha){
      final_prediction = mean_es
    }else{
      final_prediction = trajectory
    }
    if(model=="apc"){
      apc_results$predicted[apc_results$capability==capability_name] = final_prediction
    }else{
      cpc_results$predicted[cpc_results$capability==capability_name] = final_prediction
    }
  }
}

apc_results$model = "APC"
cpc_results$model = "CPC"




# Plot effect sizes
plot_traj <- create_plot_trajectories(apc_results, "APC")
print(plot_traj)
overlapped_plot_traj <- get_overlapped_plot(apc_results, "APC")
print(overlapped_plot_traj)

plot_traj <- create_plot_trajectories(cpc_results, "CPC")
print(plot_traj)
overlapped_plot_traj <- get_overlapped_plot(cpc_results, "CPC")
print(overlapped_plot_traj)



