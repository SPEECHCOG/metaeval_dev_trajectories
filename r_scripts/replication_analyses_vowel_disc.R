# Meta-regression analyses and Funnel plot. Inclusion criteria as in (Tsuji & Cristia, 2014).
# Meta-regression as in MetaLab (https://github.com/langcog/metalab/blob/main/shinyapps/visualization/server.R)
# MetaLab data downloaded on 01.03.2021 (native contrasts) and 18.03.2021 (non-native contrasts)

library(tidyverse)
# Funnel plot & analyses
library(meta)
library(metafor)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

nat_vowels <- read.csv("Vowel_discrimination_native_r.csv")  # 145 records
nonnat_vowels <- read.csv("Vowel_discrimination_non_native_r.csv")  # 49 records

# Inclusion criteria as in Tsuji & Cristia, 2014

# The study focused on normally developing infants with at least one age group involved being 12-mo of age or less
nat_vowels <- filter(nat_vowels, infant_type!='premature')  # 143 records
nonnat_vowels <- filter(nonnat_vowels, infant_type!='premature')  # 49 records

nat_vowels <- nat_vowels %>% group_by(study_ID) %>% filter(any(mean_age_1<=12*30.42))  # 143 records
nonnat_vowels <- nonnat_vowels %>% group_by(study_ID) %>% filter(any(mean_age_1<=12*30.42))  # 45 records

# At least two age groups were assessed on the same vowel contrast
nat_vowels <- nat_vowels %>% group_by(study_ID) %>% group_by(contrast_sampa) %>% filter(length(unique(mean_age_1))>1)  # 112 records
nonnat_vowels <- nonnat_vowels %>% group_by(study_ID) %>% group_by(contrast_sampa) %>% filter(length(unique(mean_age_1))>1) # 34 records

# Infants younger than 15-mo age
nat_vowels <- filter(nat_vowels, mean_age_1<=15*30.42)  # 104 records
nonnat_vowels <- filter(nonnat_vowels, mean_age_1<=15*30.42)  # 32 records

# Outliers 
# 3 outliers study_ID (unique_row): kuhl1979 (59), marean1992 (80) and pons2012 (128)
nat_vowels$outlier <-0
nat_vowels$outlier[nat_vowels$g_calc > mean(nat_vowels$g_calc,na.rm=TRUE)+3*sd(nat_vowels$g_calc,na.rm=TRUE)|nat_vowels$g_calc < mean(nat_vowels$g_calc,na.rm=TRUE)-3*sd(nat_vowels$g_calc,na.rm=TRUE)] <- 1
table(nat_vowels$outlier)  

# 1 outlier study_ID (unique_row): cardillo2010 (12)
nonnat_vowels$outlier <-0
nonnat_vowels$outlier[nonnat_vowels$g_calc > mean(nonnat_vowels$g_calc,na.rm=TRUE)+3*sd(nonnat_vowels$g_calc,na.rm=TRUE)|nonnat_vowels$g_calc < mean(nonnat_vowels$g_calc,na.rm=TRUE)-3*sd(nonnat_vowels$g_calc,na.rm=TRUE)] <-1
table(nonnat_vowels$outlier)  

# Funnel plots
get_funnel_plot <- function(data){
  rma_data <- rma(yi=g_calc, vi=g_var_calc, slab = make.unique(short_cite), data=data, method='REML') # basic model
  funnel(rma_data, level=c(90, 95, 99), shade=c("white", "gray75", "gray55"), digits=2L, xlab = "Effect Size", cex.lab=1.3, cex.axis=1.3)
  print(rma_data)
  
  # Asymmetry tests
  # Rank test (Kendall's test): if significant -> significant asymmetry (presence of small-study effects)
  # null-hypothesis of no small-study effects is rejected
  # Egger's test: if intercept significantly different from zero, then there is funnel plot asymmetry 
  
  # Kendall's test
  print(ranktest(rma_data))
  # Egger's test
  reg_line_data <- regtest(rma_data)
  print(reg_line_data)
  
  se <- seq(0, 1.8, length=100)
  lines(coef(reg_line_data$fit)[1] + coef(reg_line_data$fit)[2]*se, se, lwd=1, col="red", lty=2)  # add line to funnel plot
}

# Native vowels
get_funnel_plot(nat_vowels)
# Random-Effects Model (k = 104; tau^2 estimator: REML)
# 
# tau^2 (estimated amount of total heterogeneity): 0.1270 (SE = 0.0280)
# tau (square root of estimated tau^2 value):      0.3564
# I^2 (total heterogeneity / total variability):   67.12%
# H^2 (total variability / sampling variability):  3.04
# 
# Test for Heterogeneity:
#   Q(df = 103) = 302.9526, p-val < .0001
# 
# Model Results:
#   
# estimate      se    zval    pval   ci.lb   ci.ub 
#   0.4220  0.0445  9.4900  <.0001  0.3348  0.5091  *** 
#   
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# 
# Rank Correlation Test for Funnel Plot Asymmetry
# 
# Kendall's tau = 0.4190, p < .0001
# 
# 
# Regression Test for Funnel Plot Asymmetry
# 
# Model:     mixed-effects meta-regression model
# Predictor: standard error
# 
# Test for Funnel Plot Asymmetry: z = 7.3175, p < .0001
# Limit Estimate (as sei -> 0):   b = -0.5175 (CI: -0.7755, -0.2594)

# Non-native vowels
get_funnel_plot(nonnat_vowels)
# Random-Effects Model (k = 32; tau^2 estimator: REML)
# 
# tau^2 (estimated amount of total heterogeneity): 0.4684 (SE = 0.1351)
# tau (square root of estimated tau^2 value):      0.6844
# I^2 (total heterogeneity / total variability):   89.97%
# H^2 (total variability / sampling variability):  9.97
# 
# Test for Heterogeneity:
#   Q(df = 31) = 169.2571, p-val < .0001
# 
# Model Results:
#   
# estimate      se    zval    pval   ci.lb   ci.ub 
#   0.4649  0.1291  3.6019  0.0003  0.2119  0.7178  *** 
#   
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# 
# Rank Correlation Test for Funnel Plot Asymmetry
# 
# Kendall's tau = 0.2661, p = 0.0328
# 
# 
# Regression Test for Funnel Plot Asymmetry
# 
# Model:     mixed-effects meta-regression model
# Predictor: standard error
# 
# Test for Funnel Plot Asymmetry: z = 6.3704, p < .0001
# Limit Estimate (as sei -> 0):   b = -1.5966 (CI: -2.2376, -0.9556)

# Moderators analysis and developmental curve
# remove outliers for analyses (MetaLAB standard practices)
nat_vowels_mod <- filter(nat_vowels, outlier==0)
nonnat_vowels_mod <- filter(nonnat_vowels, outlier==0)

# Native vowels
mod_nat <- rma.mv(g_calc~mean_age_1+response_mode+exposure_phase, V=g_var_calc, slab = make.unique(short_cite), 
                  data=nat_vowels_mod, method='REML', random = ~1|short_cite/same_infant_calc/unique_row)
# Multivariate Meta-Analysis Model (k = 101; method: REML)
# 
# Variance Components:
#   
#             estim    sqrt  nlvls  fixed                                  factor 
# sigma^2.1  0.0589  0.2427     26     no                              short_cite 
# sigma^2.2  0.0000  0.0000     98     no             short_cite/same_infant_calc 
# sigma^2.3  0.0524  0.2289    101     no  short_cite/same_infant_calc/unique_row 
# 
# Test for Residual Heterogeneity:
#   QE(df = 93) = 222.0601, p-val < .0001
# 
# Test of Moderators (coefficients 2:8):
#   QM(df = 7) = 17.7100, p-val = 0.0133
# 
# Model Results:
#   
#                                estimate      se     zval    pval    ci.lb    ci.ub 
# intrcpt                          0.8854  0.2106   4.2033  <.0001   0.4725   1.2982  *** 
# mean_age_1                       0.0006  0.0005   1.1961  0.2317  -0.0004   0.0015      
# response_modeEEG                -0.5211  0.4213  -1.2369  0.2161  -1.3469   0.3047      
# response_modeeye-tracking       -0.7165  0.2841  -2.5223  0.0117  -1.2733  -0.1597    * 
# response_modeNIRS               -0.4522  0.3688  -1.2261  0.2201  -1.1750   0.2706      
# exposure_phasefamiliarization    0.1477  0.3102   0.4761  0.6340  -0.4603   0.7557      
# exposure_phasehabituation       -0.1159  0.2730  -0.4245  0.6712  -0.6508   0.4191      
# exposure_phasetest_only          0.0847  0.3288   0.2577  0.7967  -0.5597   0.7291      
# 
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Non-native vowels
mod_nonnat <- rma.mv(g_calc~mean_age_1+response_mode+exposure_phase, V=g_var_calc, slab = make.unique(short_cite), 
                  data=nonnat_vowels_mod, method='REML', random = ~1|short_cite/same_infant_calc/unique_row)
# Multivariate Meta-Analysis Model (k = 31; method: REML)
# 
# Variance Components:
#   
#             estim    sqrt  nlvls  fixed                                  factor 
# sigma^2.1  0.6541  0.8088     10     no                              short_cite 
# sigma^2.2  0.0317  0.1780     22     no             short_cite/same_infant_calc 
# sigma^2.3  0.0000  0.0000     31     no  short_cite/same_infant_calc/unique_row 
# 
# Test for Residual Heterogeneity:
#   QE(df = 23) = 59.5124, p-val < .0001
# 
# Test of Moderators (coefficients 2:8):
#   QM(df = 7) = 10.0866, p-val = 0.1837
# 
# Model Results:
#   
#                                estimate      se     zval    pval    ci.lb   ci.ub 
# intrcpt                          2.0184  0.5247   3.8464  0.0001   0.9899  3.0468  *** 
# mean_age_1                      -0.0007  0.0008  -0.9156  0.3599  -0.0022  0.0008      
# response_modeEEG                -1.2844  1.5593  -0.8237  0.4101  -4.3407  1.7718      
# response_modeeye-tracking       -1.5360  1.0005  -1.5352  0.1247  -3.4970  0.4250      
# response_modeNIRS               -1.3169  1.4363  -0.9169  0.3592  -4.1320  1.4981      
# exposure_phasefamiliarization   -0.2083  1.1958  -0.1742  0.8617  -2.5520  2.1354      
# exposure_phasehabituation       -0.0444  1.1839  -0.0375  0.9701  -2.3649  2.2760      
# exposure_phasetest_only         -0.4758  1.2075  -0.3940  0.6936  -2.8424  1.8908      
# 
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Developmental curve as in MetaLAB (effect as a function of age)

get_developmental_curve <- function(data, mean_es, lb_es, ub_es, lb_y, ub_y){
  data <- data %>% mutate(mean_age_mo = mean_age_1/30.42)
  p <- data %>%
    ggplot(aes(x=mean_age_mo, y=g_calc)) +
    geom_point(aes(size = n_1), alpha = .3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
    scale_size_continuous(guide="none") +
    xlab("\nMean Age (Months)") + 
    ylab("Effect Size\n")
  
  if(is.na(mean_es)){
    # age moderator is significant
    p <- p + geom_smooth(aes(weight=1/g_var_calc), method = "lm", colour="blue", 
                         size=0.9, se=TRUE, fill="grey70")
  } else{
    # age moderator is non-significant. effect size does not change with age
    p <- p +annotate('ribbon', x = c(-Inf, Inf), ymin = lb_es, ymax = ub_es, 
               alpha = 0.5, fill = 'grey70') +
      geom_hline(yintercept = mean_es, linetype='solid', colour="blue", size=0.9)
  }
  
  if(!is.na(lb_y) & !is.na(ub_y)){
    p <- p + scale_y_continuous(expand = c(0, 0), limits = c(lb_y, ub_y),
                                breaks = seq(lb_y, ub_y, 1)) +
      xlim(0, 15) +
      coord_cartesian(clip = "off")
  }
  
  p + theme(legend.position = "left", text = element_text(size=18), 
            axis.line = element_line(color='black', size=1)) +
    labs(colour="")
}

# Age moderator was not significant, then effect size does not change as a function
# of age
rma_nat <- rma(yi=g_calc, vi=g_var_calc, slab = make.unique(short_cite), data=nat_vowels, method='REML')
mean_es_nat = rma_nat$b[1,]
mean_es_nat_ci.lb = rma_nat$ci.lb
mean_es_nat_ci.ub = rma_nat$ci.ub

get_developmental_curve(nat_vowels_mod, mean_es_nat, mean_es_nat_ci.lb, mean_es_nat_ci.ub, -1.5, 2.5)

rma_nonnat <- rma(yi=g_calc, vi=g_var_calc, slab = make.unique(short_cite), data=nonnat_vowels, method='REML')
mean_es_nonnat = rma_nonnat$b[1,]
mean_es_nonnat_ci.lb = rma_nonnat$ci.lb
mean_es_nonnat_ci.ub = rma_nonnat$ci.ub

get_developmental_curve(nonnat_vowels_mod, mean_es_nonnat, mean_es_nonnat_ci.lb, mean_es_nonnat_ci.ub, -1, 3.2)


