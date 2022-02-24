# Replication analysis "Quantifying Sources of Variability in Infancy Research 
# Using the Infant-Directed-Speech Preference", Frank, Alcock, Arias-Trejo et al., 2020
# Part of the code has been extracted from https://github.com/manybabies/mb1-analysis-public/blob/master/paper/mb1-paper.Rmd

library("papaja")
library(ggthemes)
library(lme4)
library(tidyverse)
library(here)
library(knitr)
library(kableExtra)
library(ggpubr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Basic meta-analysis to obtain mean effect size

d <- read_csv("MB1_data/03_data_trial_main.csv", 
              na = c("NA", "N/A")) %>%
  mutate(method = case_when(
    method == "singlescreen" ~ "Central fixation",
    method == "eyetracking" ~ "Eye tracking",
    method == "hpp" ~ "HPP",
    TRUE ~ method)) 
diffs <- read_csv("MB1_data/03_data_diff_main.csv",
                  na = c("NA", "N/A")) %>%
  mutate(method = case_when(
    method == "singlescreen" ~ "Central fixation",
    method == "eyetracking" ~ "Eye tracking",
    method == "hpp" ~ "HPP",
    TRUE ~ method)) 

ordered_ages <- c("3-6 mo", "6-9 mo", "9-12 mo", "12-15 mo")
d$age_group <- fct_relevel(d$age_group, ordered_ages)
diffs$age_group <- fct_relevel(diffs$age_group, ordered_ages)
source("MB1_data/helper/ma_helper.R")
ages <- d %>%
  group_by(lab, age_group, method, nae, subid) %>%
  summarise(age_mo = mean(age_mo)) %>%
  summarise(age_mo = mean(age_mo))
ds_zt <- diffs %>%
  group_by(lab, age_group, method, nae, subid) %>%
  summarise(d = mean(diff, na.rm = TRUE)) %>%
  group_by(lab, age_group, method, nae) %>%
  summarise(d_z = mean(d, na.rm = TRUE) / sd(d, na.rm = TRUE), 
            n = length(unique(subid)), 
            d_z_var = d_var_calc(n, d_z)) %>%
  filter(n >= 10) %>%
  left_join(ages) %>%
  filter(!is.na(d_z)) # CHECK THIS 

# For the computational evaluation, we focus only on the North American English subset
ds_zt_nae <- filter(ds_zt, nae==TRUE)

# Random-Effect Model 
intercept_mod <- metafor::rma(d_z ~ 1, 
                              vi = d_z_var, slab = lab, data = ds_zt_nae, 
                              method = "REML") 

# Random-Effects Model (k = 49; tau^2 estimator: REML)
# 
# tau^2 (estimated amount of total heterogeneity): 0.0198 (SE = 0.0243)
# tau (square root of estimated tau^2 value):      0.1407
# I^2 (total heterogeneity / total variability):   16.22%
# H^2 (total variability / sampling variability):  1.19
# 
# Test for Heterogeneity:
#   Q(df = 48) = 58.6010, p-val = 0.1405
# 
# Model Results:
#   
# estimate      se    zval    pval   ci.lb   ci.ub 
#   0.4319  0.0503  8.5849  <.0001  0.3333  0.5305  *** 
#   
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Funnel plot
metafor::funnel(intercept_mod, level=c(90, 95, 99), shade=c("white", "gray75", "gray55"), digits=2L, xlab = "Effect Size", cex.lab=1.3, cex.axis=1.3)
# Eeger's test
reg_nat <- metafor::regtest(intercept_mod)
# se <- seq(0, 1.8, length=100)
# lines(coef(reg_nat$fit)[1] + coef(reg_nat$fit)[2]*se, se, lwd=1, col="red", lty=2)

# Native vs Non-native language
ds_zt_nae$english <- factor(ds_zt$nae, levels = c(TRUE, FALSE), 
                        labels = c("North American English", "Non-North American English")) 

# Linear Mixed-Effect model as in paper
library(lmerTest)

# NAE subset
d_nae <- filter(d, nae==TRUE)
d_lmer <- d_nae %>%
  filter(trial_type != "train") %>%
  mutate(log_lt = log(looking_time),
         age_mo = scale(age_mo, scale = FALSE),
         trial_num = trial_num, 
         item = paste0(stimulus_num, trial_type)) %>%
  filter(!is.na(log_lt), !is.infinite(log_lt))

mod_lmer <- lmer(log_lt ~ trial_type * method +
                   trial_type * trial_num +
                   age_mo * trial_num +
                   trial_type * age_mo * nae +
                   (1 | subid_unique) +
                   (1 | item) + 
                   (1 | lab), 
                 data = d_lmer)

coefs <- summary(mod_lmer)$coef %>%
  as_tibble %>%
  mutate_at(c("Estimate","Std. Error","df", "t value", "Pr(>|t|)"), 
            function (x) signif(x, digits = 3)) %>%
  rename(SE = `Std. Error`, 
         t = `t value`,
         p = `Pr(>|t|)`) %>%
  select(-df)


rownames(coefs) <- c("Intercept", "IDS", "Eye-tracking", "HPP", 
                     "Trial #", "Age", "IDS * Eye-tracking", 
                     "IDS * HPP", 
                     "IDS * Trial #", "Trial # * Age", "IDS * Age")
papaja::apa_table(coefs, 
                  caption = "Coefficient estimates from a linear mixed effects model predicting log looking time.", 
                  format.args = list(digits = 3),
                  col.names =c("","Estimate","$SE$","$t$","$p$"),
                  align=c("l","l","c","c","c"))


# Developmental curve 

# Basic linear model d ~ 1 + age
# For reference to comparison with models (computational evaluation paper)
ggplot(ds_zt_nae, 
       aes(x = age_mo, y = d_z)) + 
  geom_point(aes(size = n), alpha = .3) + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  geom_smooth(method = "lm", colour="blue", size=0.9, se=TRUE, fill="grey70") + 
  scale_size_continuous(guide = "none") +
  scale_y_continuous(expand = c(0, 0), limits = c(-1, 2.3),
                       breaks = seq(-1, 2.3, 1)) +
  coord_cartesian(clip = "off") +
  xlab("\nMean Age (Months)") +
  ylab("Effect Size\n") + 
  theme(legend.position = "right", text = element_text(size=18), 
        axis.line = element_line(color='black', size=1)) +
  labs(colour="")















