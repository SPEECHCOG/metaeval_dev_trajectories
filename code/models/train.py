"""
@date 08.02.2022

It trains a model according to the configuration given
"""
import argparse
import sys

from core.apc import APCModel
from core.cpc import CPCModel
from read_configuration import read_configuration_json


def train(config_path: str) -> None:
    """
    Train a neural network model using the configuration parameters provided
    :param config_path: the path to the JSON configuration file.
    :return: The trained model is saved in a h5 file
    """

    # read configuration file
    config = read_configuration_json(config_path)['training']

    # Use correct model
    model_type = config['model']['type']

    if model_type == 'apc':
        model = APCModel()
    elif model_type == 'cpc':
        model = CPCModel()
    else:
        raise Exception('The model type "%s" is not supported' % model_type)

    model.load_training_configuration(config)
    model.train()

    print('Training of model "%s" finished' % model_type)


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for training a model from the ones available in the python '
                                                 'module using a JSON configuration file. \nUsage: python train.py '
                                                 '--config path_JSON_configuration_file')
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    # Train model
    train(args.config)
    sys.exit(0)
