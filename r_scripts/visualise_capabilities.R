# It plots the developmental trajectories of (APC and CPC) models for the two 
# linguistic capabilities (IDS preference and Vowel discrimination)
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
  models_subset_data <- models_data
  if(cap=="IDS preference"){
    models_subset_data <- models_subset_data %>% filter(days>=130)
  }else if(cap=="Non-native discrimination"){
    models_subset_data <- models_subset_data %>% filter(days>=106)
  }
  
  colours_plot = c("APC" = "#7629A0", "CPC" = "#E27000")
  labels_plot = c(expression(APC[NOV]), expression(CPC[NOV]))
  
  overlapped_plot <- ggplot(infants_data, aes(x = days, y = ds)) + 
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey28") +
    
    scale_size_continuous(guide = "none") +
    coord_cartesian(clip = "off") +
    xlab("Simulated model age / real infant age (days)") +
    ylab("Effect Size") + 
    
    geom_point(size=1.5, data=models_data, aes(y=d, x=days, 
                                               colour=model)) +
    guides("shape"="none") +
    #scale_shape_manual(values =c("batch"=1, "epoch"=20), )+
    geom_line(size=1, data=models_data, aes(y=predicted, x=days, colour=model)) + 
    
    #ggtitle(cap) + 
    
    theme(legend.position = c(0.7,0.85), 
          legend.text=element_text(family="LM Roman 10", size=16),
          legend.title=element_blank(), 
          text = element_text(family="LM Roman 10", size=18), 
          axis.line = element_line(color='black', size=1), 
          plot.title = element_text(hjust = 0.5))
  
  if(cap=="IDS preference"){
    # IDS 
    overlapped_plot <- overlapped_plot + geom_smooth(method = "lm", 
                                                     size=1, se=TRUE, 
                                                     xseq=130:250, 
                                                     colour="#00BA38",
                                                     fill="#00BA38") +
      #geom_smooth(size=1, se=FALSE, method="lm", data=models_subset_data, 
      #            aes(y=d, x=days, group=model), colour="gray4") +
      scale_y_continuous(expand = c(0, 0), limits = c(-1, 1.5),
                         breaks = seq(-1, 1.5, 0.5))
    colours_plot <- c(colours_plot, "IDS preference" = "#00BA38")
    labels_plot <- c(labels_plot, "Infants")
    
  }else{
    # Vowel discrimination
    overlapped_plot <- overlapped_plot + 
      geom_line(size=1, aes(x = days, y = ds, colour=capability), 
                data=infants_data)
    if(cap=="Native discrimination"){
      overlapped_plot <- overlapped_plot +
        annotate('ribbon', x = c(3, 166), ymin = mean_es_nat_ci.lb, 
                 ymax = mean_es_nat_ci.ub, 
                 alpha = 0.5, fill=alpha("#F8766D", alpha=0.2)) +
        scale_y_continuous(expand = c(0, 0), limits = c(-0.5, 4.3),
                           breaks = seq(-0.5, 4.3, 0.5)) 
        #ggtitle("Native vowel discrimination")
      colours_plot <- c(colours_plot, "Native discrimination" = "#F8766D")
      labels_plot <- c(labels_plot, "Infants") 
    }else{
      overlapped_plot <- overlapped_plot +
        annotate('ribbon', x = c(106, 166), ymin = mean_es_nonnat_ci.lb, 
                 ymax = mean_es_nonnat_ci.ub, 
                 alpha = 0.5, fill=alpha("#619CFF", alpha=0.8))+
        #geom_smooth(size=1, se=FALSE, method="lm", data=models_subset_data, 
        #            aes(y=d, x=days, group=model), colour="gray3") + 
        scale_y_continuous(expand = c(0, 0), limits = c(-0.5, 2.5),
                           breaks = seq(-0.5, 2.5, 0.5)) 
        #ggtitle("Non-native vowel discrimination")
      
      colours_plot <- c(colours_plot, "Non-native discrimination" = "#619CFF")
      labels_plot <- c(labels_plot, "Infants")
    }
  }
  
  # Colouring 
  overlapped_plot <- overlapped_plot + 
    scale_color_manual(values = colours_plot, labels=labels_plot)+
    scale_fill_manual(values = colours_plot, labels=labels_plot)
  
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

final_results <- rbind(apc_results, cpc_results)

write.csv(final_results, "results.csv")

# Plot effect sizes
print(get_overlapped_plot_per_cap(final_results, "IDS preference"))
print(get_overlapped_plot_per_cap(final_results, "Native discrimination"))
print(get_overlapped_plot_per_cap(final_results, "Non-native discrimination"))




