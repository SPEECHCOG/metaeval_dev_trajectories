"""
@date 08.02.2022
New models should inherit from this class so training and prediction can be done in the same way for
all the models and independently.
"""
import copy
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


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
        self.model_type = model_config['type']
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
        self.features = model_config['num_features']
        self.sample_size = model_config['sample_size']  # frames per sample
        self.input_shape = (self.sample_size, self.features)  # input features dim (samples x frames x features)
        self.path_train_data = config['input_features']['train']
        self.path_validation_data = config['input_features']['validation']

        # Create folder structure where to save the model/checkpoints
        self.full_path_output_folder = os.path.join(self.output_folder, self.name, self.features_folder_name)
        os.makedirs(self.full_path_output_folder, exist_ok=True)
        self.logs_folder_path = os.path.join(self.full_path_output_folder, 'logs/')
        os.makedirs(self.logs_folder_path, exist_ok=True)

    @abstractmethod
    def train(self) -> List[Union[TensorBoard, ModelCheckpoint, EarlyStopping]]:
        """
        It sets model's path and callbacks according to configuration provided.
        :return: trained model
        """
        # Model file name for checkpoint and log
        model_file_name = os.path.join(self.full_path_output_folder, f'{self.model_type}_{self.language}'
                                                                     f'{datetime.now().strftime("_%Y_%m_%d-%H_%M")}')

        # log
        self.write_log(f'{model_file_name}.txt')

        if self.save_untrained:
            folder_path, file_name = os.path.split(model_file_name)
            self.model.save(f'{folder_path}/untrained_{file_name}.h5')

        # Callbacks for training
        callbacks = []
        # Tensorboard
        log_dir = os.path.join(self.logs_folder_path, self.language, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)
        callbacks.append(tensorboard)

        if self.checkpoint_period is not None:
            checkpoint = ModelCheckpoint(f'{model_file_name}_epoch-{{epoch:d}}.h5', monitor='val_loss',
                                         mode='min', verbose=1, period=self.checkpoint_period)
            callbacks.append(checkpoint)
        else:
            # Adding early stop based on validation loss and saving best model for later prediction
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
            checkpoint = ModelCheckpoint(model_file_name + '.h5', monitor='val_loss', mode='min', verbose=1,
                                         save_best_only=True)
            callbacks.append(early_stop)
            callbacks.append(checkpoint)

        return callbacks





