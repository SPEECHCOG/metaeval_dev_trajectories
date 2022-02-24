# Developmental trajectories of IDS preference and Vowel discrimination 
# obtained using meta-analysis.

source("replication_analyses_ids.R")
source("replication_analyses_vowel_disc.R")

options(dplyr.summarise.inform = FALSE)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

ids_df <- data.frame(
  ds = ds_zt_nae$d_z * (1 - (3/(4*(ds_zt_nae$n*2) - 9))),  # Hedges' correction
  days = ds_zt_nae$age_mo*30.4375, # according to scale used in the original data
  capability="IDS preference"
)

ids_df <- ids_df %>% filter(days<=166)

vd_nat_df <- data.frame(
  ds = rep(mean_es_nat, 166),
  days = 1:166,
  capability="Native discrimination"
)

vd_nonnat_df <- data.frame(
  ds = rep(mean_es_nonnat, 166),
  days = 1:166,
  capability="Non-native discrimination"
)

all_capabilities <- rbind(ids_df, vd_nat_df, vd_nonnat_df)

ggplot(all_capabilities, 
       aes(x = days, y = ds, colour=capability)) +  
  scale_color_manual(values = c("IDS preference" = "#F8766D",
                                "Native discrimination" = "#619CFF",
                                "Non-native discrimination" = "#00BA38")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  geom_smooth(method = "lm", size=0.9, se=TRUE, fill="grey70") + 
  scale_size_continuous(guide = "none") +
  scale_y_continuous(expand = c(0, 0), limits = c(-1, 3),
                     breaks = seq(-1, 3, 0.5)) +
  coord_cartesian(clip = "off") +
  xlab("\nMean age (days)") +
  ylab("Effect Size\n") + 
  title("Infants") + 
  theme(axis.line = element_line(color='black', size=1), plot.title = element_text(hjust = 0.5))
  #theme(legend.position = "right", text = element_text(size=18), 
  #      axis.line = element_line(color='black', size=1)) +
  #labs(colour="")
