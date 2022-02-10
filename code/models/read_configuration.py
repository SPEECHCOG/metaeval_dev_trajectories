"""
@date 08.02.2022

It reads a configuration in the JSON format for training. It validates the fields of the configurations.
"""
import json
import os
import pathlib
from typing import Optional, List, Union, Tuple

import h5py
import numpy as np

__docformat__ = ['reStructuredText']
__all__ = ['load_all_training_data', 'load_training_file', 'load_epoch_training_data', 'read_configuration_json']

# Fields of configuration (name, type, default value, required?).
TRAINING_CONFIG = [
    ('output_path', str, None, True),
    ('features_folder_name', str, None, True),
    ('language', str, 'mandarin', False),
    ('model', dict, None, True),
    ('input_features', dict, None, False),
    ('save_untrained', bool, False, False),
    ('checkpoint_period', int, None, False),
    ('data_schedule', str, 'all', True)
]

TRAINING_MODEL_CONFIG = [
    ('name', str, None, True),
    ('type', str, None, True),
    ('epochs', int, 100, False),
    ('early_stop_epochs', int, 50, False),
    ('batch_size', int, 32, False),
    ('latent_dimension', int, None, True),
    ('apc', dict, None, False),
    ('input_attention', bool, False, False)
]

INPUT_FEATURES_CONFIG = [
    ('train', str, None, True),
    ('validation', str, None, True)
]

APC_CONFIG = [
    ('prenet', bool, False, False),
    ('prenet_layers', int, 0, False),
    ('prenet_dropout', float, 0, False),
    ('prenet_units', int, 0, False),
    ('rnn_layers', int, 3, False),
    ('rnn_dropout', float, 0, False),
    ('rnn_units', int, 512, False),
    ('residual', bool, True, False),
    ('learning_rate', float, 0.001, False),
    ('steps_shift', int, 5, False)
]

CPC_CONFIG = [
    ("encoder_layers", int, 5, False),
    ("encoder_units", int, 512, False),
    ("encoder_dropout", float, 0.2, False),
    ("gru_units", int, 256, False),
    ("dropout", float, 0.2, False),
    ("neg", int, 10, False),
    ("steps", int, 12, False),
    ("learning_rate", float, 0.001, False)
]

DATA_SCHEDULES = ["all", "epoch"]  # TODO incremental


def _validate_fields(config, config_definition):
    """
    It validates a configuration against the required information (config_definition). It a required field is not
    provided, or the type of the field is not as the required an exception is raised. If non-required fields are not
    given, they are set to the default value
    :param config: a dictionary with the configuration
    :param config_definition: a list of fields definitions
    :return: It modifies the config dictionary
    """
    current_fields = list(config.keys())

    for field in config_definition:
        if field[0] not in current_fields:
            if field[3]:
                # It is a required field
                raise Exception('The training configuration is missing required field "%s"' % field[0])
            else:
                # Create field with default value
                config[field[0]] = field[2]
        else:
            # The field was provided.
            if type(config[field[0]]) != field[1]:
                raise Exception('The field "%s" should be of type %s' % (field[0], str(field[1])))


def _validate_training(config):
    """
    It validates the training configuration data is in the right format, that paths exist, and assigns default
    values if non-required fields are not given.
    :param config: a dictionary with the configuration
    :return: The configuration with custom and default values where needed if the validation is satisfactory,
             otherwise an exception will be raised.
    """
    # Validate training fields
    _validate_fields(config['training'], TRAINING_CONFIG)
    # Validate model fields
    _validate_fields(config['training']['model'], TRAINING_MODEL_CONFIG)

    # Validate data schedule
    if config['training']['data_schedule'] not in DATA_SCHEDULES:
        raise Exception('Only ' + ', '.join(DATA_SCHEDULES) + ' are supported.')

    # Validate training files fields
    _validate_fields(config['training']['input_features'], INPUT_FEATURES_CONFIG)
    for path in [config['training']['input_features']['train'], config['training']['input_features']['validation']]:
        if not os.path.exists(path):
            raise Exception('The file does not exist: %s' % path)

    # Validate models configuration
    # apc
    if config['training']['model']['type'] == 'apc':
        if not config['training']['model']['apc']:
            # The model is apc but there were no parameters, we need to create default parameters
            config['training']['model']['apc'] = {}
        # validate parameters for apc model
        _validate_fields(config['training']['model']['apc'], APC_CONFIG)
    elif config['training']['model']['type'] == 'cpc':
        if not config['training']['model']['cpc']:
            config['training']['model']['cpc'] = {}
        _validate_fields(config['training']['model']['cpc'], CPC_CONFIG)

    return config


def _shift_features(x_train: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    It shifts an array according to the given steps. Then it adjusts size of input and output features arrays.
    :param x_train: array with dim(samples x frames x features)
    :param steps: total of steps for shifting
    :return: input and output where output features array is shifted
    """
    y_train = None

    sample_length = x_train.shape[1]
    n_feats = x_train.shape[-1]

    x_train_flat = x_train.reshape((-1, n_feats))
    x_train = x_train_flat[:-steps, :]
    y_train = x_train_flat[steps:, :]

    total_samples_tr = int(x_train.shape[0] / sample_length)
    total_frames_tr = total_samples_tr * sample_length
    x_train = x_train[:total_frames_tr, :]
    y_train = y_train[:total_frames_tr, :]

    x_train = x_train.reshape((total_samples_tr, sample_length, -1))
    y_train = y_train.reshape((total_samples_tr, sample_length, -1))

    return x_train, y_train


def load_all_training_data(path: Union[str, pathlib.Path], shift: Optional[bool] = False,
                           steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    It reads the path where input features are located. The path can be a directory or a h5py file.
    :param path: path of directory with h5py files or path of h5py file with input features.
    :param shift: it states if a shift should be performed.
    :param steps: number of steps to shift in APC.
    :return: input and expected output features.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        features = []
        for file_path in path.iterdir():
            with h5py.File(file_path, 'r') as h5py_file:
                features.append(np.array(h5py_file['data']))
        x_train = np.concatenate(features)
    else:
        with h5py.File(path, 'r') as h5py_file:
            x_train = np.array(h5py_file['data'])

    y_train = None

    if shift:
        assert steps is not None
        x_train, y_train = _shift_features(x_train, steps)

    return x_train, y_train


def load_training_file(path: Union[pathlib.Path, str], shift: Optional[bool] = False,
                       steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    It loads a h5py file with input features and return the array.
    :param path: path of h5py file
    :param shift: It indicates if features should be shifted.
    :param steps: Number of steps in which the features should be shifted.
    :return: array with input features
    """
    with h5py.File(path, 'r') as h5py_file:
        x_train = np.array(h5py_file['data'])

    y_train = None

    if shift:
        assert steps is not None
        x_train, y_train = _shift_features(x_train, steps)

    return x_train, y_train


def load_epoch_training_data(path: Union[pathlib.Path, str]) -> List[pathlib.Path]:
    """
    It returns the files located in path sorted by size (from largest to smallest).
    :param path: a directory path
    :return: list of files within the directory
    """
    path = pathlib.Path(path)
    if path.is_dir():
        h5py_files = sorted([h5py_file for h5py_file in path.iterdir()], key=lambda x: os.stat(x).st_size, reverse=True)
    else:
        raise Exception(f"Input features path should be a directory: {path}")
    return h5py_files


def read_configuration_json(json_path):
    """
    It reads a JSON file that contains the configuration for both training and prediction.
    The validation of training configurations. It returns the validated configuration.
    An exception is raised if the path does not exists.
    :param json_path: string specifying the path where the JSON configuration file is located
    :return: a dictionary with the configuration parameters
    """
    with open(json_path) as f:
        configuration = json.load(f)
        if 'training' not in configuration:
            raise Exception('The configuration should have "training" field for training configuration.')
        else:
            configuration = _validate_training(configuration)
        return configuration
