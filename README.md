# Computational modelling of  developmental trajectories of infant language skills with meta-analytic reference data

Computational modelling of early language acquisition aims to 
investigate how different factors lead to infants' language behaviours. 
Building such models also requires the development of evaluation 
methodologies to assess and compare model validity. A critical 
component of the evaluation is the selection of the human reference 
data that best reflects current knowledge of infants' language skills. 
In addition, the evaluation would be ideally carried as a function of 
infant's developmental stage, as infants' language skills are in 
constant change throughout childhood. This work proposes the use of 
meta-analytic data across numerous empirical studies to derive 
developmental trajectories of infant language skills, and a methodology 
to evaluate models against these trajectories. We exemplify the use of 
the proposed approach by comparing two self-supervised representation 
learning algorithms, Autoregressive Predictive Coding (APC) and 
Contrastive Predictive Coding (CPC), with infants’ behaviour for 
infant-directed speech preference and vowel discrimination capabilities 
as a function of infant age and model language experience. Results 
demonstrate how our approach allows systematic comparison of models to 
infant behaviour across the developmental timeline. Neither of the two 
models trained with English speech simultaneously reach compatible 
trajectories for the two capabilities.


## Requirements

This repository hosts the code that replicates the results reported
in "Computational modelling of  developmental trajectories of infant 
language skills with meta-analytic reference data"

The code for predictive coding models was developed usng Python 3.8 and
TensorFlow 2.1.0. The code for analysing effect sizes was developed
using R version 4.1.1

## Folder structure

* `python_scripts` contains the code for training APC and CPC models

* `r_scripts` contains the code to calculate the effect sizes from the 
csv files (output by each capability see 
[computational tests](https://github.com/SPEECHCOG/metaeval_experiments)) 
and plot the developmental trajectories.

  * `plots` developmental trajectory plots for IDS preference and vowel
  discrimination (native and non-native).

  * `test_results` measurements (validation loss or DTW distance) for each 
  model, checkpoint, capability and trial. 


## Contact
If you find any issue please report it on the 
[issues section](https://github.com/SPEECHCOG/metaeval_dev_trajectories/issues) 
in this repository. Further comments can be sent to 
`maria <dot> cruzblandon <at> tuni <dot> fi`


