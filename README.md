# Computational modelling of  developmental trajectories of infant language skills with meta-analytic reference data

## Requirements

The code for predictive coding models was developed usng Python 3.8 and
TensorFlow 2.1.0

To execute the python code  the dependencies are specified in 
`./requirements.yml` for conda the environment. To create the environment 
run:

```
conda env create -f requirements.yml
```

use `./requirements_across_platforms.yml` if you find problems with 
`requirements.yml`

Then you can activate the conda environment for excuting the code

```
conda activate metaeval_env
```


## Folder structure

* `python_scripts` contains the code for training APC and CPC models
and preprocess input features.

# Preprocess input data

Edit `preprocess_input_data.json` file to match with the path of the 
data in your computer. You can find a description of the type of data 
for each field in the `config_template_preprocess_data.json` file.

For our experiments, we used 13 MFCCs + 13 delta + 13 delta delta without
normalisation (`"cmvn": false` in the configuration file). We shuffled the 
files between LibriSpeech and SpokenCOCO and split the data into chunks of
100 h.

```
cd python_scripts
python preprocess_training_data.py --config preprocess_input_data.json

```

# Train model
Edit `config_apc.json` file to match with the paths on your computer.
This file already contains the default configuration of the model that 
replicates the results in the manuscript "Introducing meta-analysis in 
the evaluation of computational models of infant language development".
```
cd models
python train.py --config config_apc.json
```


## Contact
If you find any issue please report it on the 
[issues section](https://github.com/SPEECHCOG/metaeval_dev_trajectories/issues) 
in this repository. Further comments can be sent to 
`maria <dot> cruzblandon <at> tuni <dot> fi`


