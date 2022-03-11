# Developmental trajectories of IDS preference and Vowel discrimination 
# obtained using meta-analysis.
library(extrafont)

source("replication_analyses_ids.R")
source("replication_analyses_vowel_disc.R")

options(dplyr.summarise.inform = FALSE)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

ids_df <- data.frame(
  ds = ds_zt_nae$d_z * (1 - (3/(4*(ds_zt_nae$n*2) - 9))),  # Hedges' correction
  days = ds_zt_nae$age_mo*30.4375, # according to scale used in the original data
  capability="IDS preference"
)

vd_nat_df <- data.frame(
  ds = rep(mean_es_nat, 164),
  days = 3:166,
  capability="Native discrimination"
)

vd_nonnat_df <- data.frame(
  ds = rep(mean_es_nonnat, 61),
  days = 106:166,
  capability="Non-native discrimination"
)

all_capabilities_infants <- rbind(ids_df, vd_nat_df, vd_nonnat_df)
vd <- all_capabilities_infants %>% filter(capability != "IDS preference")


infants_plot <- 
  ggplot(all_capabilities_infants, 
       aes(x = days, y = ds, colour=capability, fill=capability)) + 
  scale_color_manual(values = c("IDS preference" = "#00BA38",
                                "Native discrimination" = "#F8766D",
                                "Non-native discrimination" = "#619CFF")) +
  scale_fill_manual(values = c("IDS preference" = "#00BA38",
                                "Native discrimination" = "#F8766D",
                                "Non-native discrimination" = "#619CFF")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  # IDS 
  geom_smooth(method = "lm", size=0.9, se=TRUE, 
              #fill=alpha("#00BA38", alpha=0.2), 
              xseq=130:166, data=ids_df, linetype="dotted") + 
  # Vowel discrimination
  annotate('ribbon', x = c(1, 166), ymin = mean_es_nat_ci.lb, 
           ymax = mean_es_nat_ci.ub, 
           alpha = 0.5, fill=alpha("#F8766D", alpha=0.2)) +
  annotate('ribbon', x = c(1, 166), ymin = mean_es_nonnat_ci.lb, 
           ymax = mean_es_nonnat_ci.ub, 
           alpha = 0.5, fill=alpha("#619CFF", alpha=0.8)) +
  geom_line(size=0.9, data=vd, linetype="dotted") +
  scale_size_continuous(guide = "none") +
  scale_y_continuous(expand = c(0, 0), limits = c(-1, 4.5),
                     breaks = seq(-1, 4.5, 0.5)) +
  coord_cartesian(clip = "off") +
  xlab("\nMean age (days)") +
  ylab("Effect Size\n") + 
  ggtitle("Infants") + 
  theme(legend.position = c(0.7,0.85), legend.text=element_text(family="LM Roman 10", size=14),
        legend.title=element_blank(), text = element_text(family="LM Roman 10", size=18), 
        axis.line = element_line(color='black', size=1), 
        plot.title = element_text(hjust = 0.5))

