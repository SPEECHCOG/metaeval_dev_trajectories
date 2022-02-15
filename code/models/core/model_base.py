"""
@date 08.02.2022
New models should inherit from this class so training and prediction can be done in the same way for
all the models and independently.
"""
import copy
import gc
import json
import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from read_configuration import load_training_file, load_all_training_data, load_epoch_training_data


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
        self.data_schedule = model_config['data_schedule']
        self.loop_epoch_data = model_config['loop_epoch_data']
        self.checkpoint_epoch_period = model_config['checkpoint_epoch_period']
        self.checkpoint_sample_period = model_config['checkpoint_sample_period']
        self.monitor_first_epoch = model_config['monitor_first_epoch']
        self.save_untrained = model_config['save_untrained']
        self.save_best = model_config['save_best']
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

        self.model = None

    @abstractmethod
    def train(self):
        """
        It sets model's path and callbacks according to configuration provided.
        :return: trained model
        """
        # Model file name for checkpoint and log
        # timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.full_path_output_folder = pathlib.Path(self.full_path_output_folder).joinpath(f'{self.model_type}_'
                                                                                           f'{self.language}_'
                                                                                           f'{timestamp}')
        os.makedirs(self.full_path_output_folder, exist_ok=True)
        model_file_name = os.path.join(self.full_path_output_folder, f'{self.model_type}_{self.language}_'
                                                                     f'{timestamp}')

        # log
        self.write_log(f'{model_file_name}.txt')

        if self.save_untrained:
            folder_path, file_name = os.path.split(model_file_name)
            self.model.save(f'{folder_path}/untrained_{file_name}.h5')

        # Callbacks for training
        callbacks = []
        callbacks_first_epoch = []

        # Tensorboard log directory
        log_dir = os.path.join(self.logs_folder_path, self.language, timestamp)
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)
        callbacks.append(tensorboard)
        # early stop based on validation loss
        if self.early_stop_epochs is not None:
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
            callbacks.append(early_stop)
        # save model with the best validation loss across all training steps.
        if self.save_best:
            checkpoint = ModelCheckpoint(model_file_name + '.h5', monitor='val_loss', mode='min', verbose=1,
                                         save_best_only=True)
        else:
            checkpoint = ModelCheckpoint(f'{model_file_name}_epoch-{{epoch:d}}.h5', monitor='val_loss',
                                         mode='min', verbose=1, period=self.checkpoint_epoch_period)
        callbacks.append(checkpoint)

        if self.data_schedule == 'epoch':
            # Only for data schedule 'epoch' monitoring the first epoch can be done.
            if self.monitor_first_epoch:
                tensorboard_fe = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0,
                                             # No. of batches
                                             update_freq=int(self.checkpoint_sample_period / self.batch_size - 1))
                callbacks_first_epoch.append(tensorboard_fe)
                if not self.save_best:
                    checkpoint_fe = ModelCheckpoint(f'{model_file_name}_epoch-{{epoch:d}}_{{batch:d}}.h5',
                                                    monitor='val_loss', mode='min', verbose=1,
                                                    save_freq=self.checkpoint_sample_period)  # No. of samples
                    callbacks_first_epoch.append(checkpoint_fe)
                else:
                    callbacks_first_epoch.append(checkpoint)

        if self.model_type == 'apc':
            x_val, y_val = load_training_file(self.path_validation_data, shift=True, steps=self.steps_shift)
        elif self.model_type == 'cpc':
            x_val, _ = load_training_file(self.path_validation_data, shift=False)
            # Create dummy prediction so that Keras does not raise an error for wrong dimension
            y_val = np.random.rand(x_val.shape[0], 1, 1)

        # Training: check input data schedule
        if self.data_schedule == 'all':
            # Train the model
            if self.model_type == 'apc':
                x_train, y_train = load_all_training_data(self.path_train_data, shift=True, steps=self.steps_shift)
            elif self.model_type == 'cpc':
                x_train, _ = load_all_training_data(self.path_train_data, shift=False)
                y_train = np.random.rand(x_train.shape[0], 1, 1)
            self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                           validation_data=(x_val, y_val), callbacks=callbacks)
        elif self.data_schedule == 'epoch':
            path_input_data = load_epoch_training_data(self.path_train_data)
            total_epochs_per_iteration = len(path_input_data)
            total_iterations = 1 if not self.loop_epoch_data else self.epochs
            best_val_loss = float('inf')
            wait = 0
            epoch = 0
            stop = False
            while epoch < total_iterations:
                for idx, path_data in enumerate(path_input_data):
                    if epoch >= total_iterations:
                        break
                    if epoch == 0 and self.monitor_first_epoch and idx == 0:
                        custom_callbacks = callbacks_first_epoch
                    else:
                        custom_callbacks = callbacks
                    if self.model_type == 'apc':
                        x_train, y_train = load_training_file(path_data, shift=True, steps=self.steps_shift)
                    elif self.model_type == 'cpc':
                        x_train, _ = load_training_file(path_data, shift=False)
                        y_train = np.random.rand(x_train.shape[0], 1, 1)
                    history = self.model.fit(x_train, y_train, epochs=epoch + 1, batch_size=self.batch_size,
                                             validation_data=(x_val, y_val), initial_epoch=epoch,
                                             callbacks=custom_callbacks)
                    del x_train
                    del y_train
                    gc.collect()
                    epoch += 1
                    # Allow early stopping when there is more than one iteration over the training data.
                    if self.early_stop_epochs is not None:
                        if epoch >= total_epochs_per_iteration:
                            if wait >= self.early_stop_epochs:
                                stop = True
                                break
                            current_val_loss = history.history['val_loss'][0]
                            if current_val_loss < best_val_loss:
                                best_val_loss = current_val_loss
                                wait = 0
                            else:
                                wait += 1
                if stop:
                    print(f'Epoch {epoch:05d}: early stopping')
                    break
