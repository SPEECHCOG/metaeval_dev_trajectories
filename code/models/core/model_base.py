"""
@date 08.02.2022
New models should inherit from this class so training and prediction can be done in the same way for
all the models and independently.
"""
import copy
import json
import os
from abc import ABC, abstractmethod


class ModelBase(ABC):
    @abstractmethod
    def write_log(self, file_path: str) -> None:
        """
        Write the log of the configuration parameters used for training an APC model
        :param file_path: path where the file will be saved
        :return: a text logfile
        """
        with open(file_path, 'w', encoding='utf8') as file:
            config = copy.deepcopy(self.configuration)
            if config['model']['type'] == 'cpc':
                if config['model']['apc']:
                    del config['model']['apc']
            elif config['model']['type'] == 'apc':
                if config['model']['cpc']:
                    del config['model']['cpc']
            file.write(json.dumps(config, indent=4))

    @abstractmethod
    def load_training_configuration(self, config: dict):
        """
        Loads the configuration parameters from the configuration dictionary and assign values to model's attributes
        :param config: a dictionary with the configuration for training
        :return: instance will have the configuration parameters
        """
        # Configuration
        self.configuration = config

        # properties from configuration file for training
        model_config = config['model']
        self.name = model_config['name']
        self.type = model_config['type']
        self.epochs = model_config['epochs']
        self.early_stop_epochs = model_config['early_stop_epochs']
        self.batch_size = model_config['batch_size']
        self.latent_dimension = model_config['latent_dimension']
        self.output_folder = config['output_path']
        self.features_folder_name = config['features_folder_name']
        self.language = config['language']
        self.input_attention = model_config['input_attention']
        self.data_schedule = config['data_schedule']
        self.checkpoint_period = config['checkpoint_period']
        self.save_untrained = config['save_untrained']

        # Create folder structure where to save the model/checkpoints
        self.full_path_output_folder = os.path.join(self.output_folder, self.name, self.features_folder_name)
        os.makedirs(self.full_path_output_folder, exist_ok=True)
        self.logs_folder_path = os.path.join(self.full_path_output_folder, 'logs/')
        os.makedirs(self.logs_folder_path, exist_ok=True)

    @abstractmethod
    def train(self):
        raise NotImplementedError('The model needs to overwrite the train method. The method should configure the '
                                  'learning process, callbacks and fit the model.')