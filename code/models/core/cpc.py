"""
@date 08.02.2022
Contrastive Predictive Coding model
[Representation Learning with Contrastive Predictive Coding]
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Input, GRU, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from core.utils import FeatureEncoder, ContrastiveLoss
from core.model_base import ModelBase
from core.apc import AttentionWeights
from read_configuration import load_training_file, load_all_training_data, load_epoch_training_data

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class CPCModel(ModelBase):

    def write_log(self, file_path: str):
        """
        Write the log of the configuration parameters used for training a CPC model
        :param file_path: path where the file will be saved
        :return: a text logfile
        """
        super(CPCModel, self).write_log(file_path)

    def load_training_configuration(self, config):
        """
        It instantiates the model architecture using the parameters from the configuration file.
        :param config: Dictionary with the configuration parameters
        :return: an instance will have the parameters from configuration and the model architecture
        """
        super(CPCModel, self).load_training_configuration(config)

        # Model architecture: Feature_Encoder -> Dropout -> GRU -> Dropout
        cpc_config = config['model']['cpc']
        # feature encoder params
        self.encoder_layers = cpc_config['encoder_layers']
        self.encoder_units = cpc_config['encoder_units']
        self.encoder_dropout = cpc_config['encoder_dropout']

        # autoregressive model params
        self.gru_units = cpc_config['gru_units']

        # contrastive loss params
        self.neg = cpc_config['neg']
        self.steps = cpc_config['steps']

        # dropout and learning rate params
        self.dropout = cpc_config['dropout']
        self.learning_rate = cpc_config['learning_rate']

        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')
        # Feature Encoder
        feature_encoder = FeatureEncoder(self.encoder_layers, self.encoder_units, self.encoder_dropout)
        encoder_features = feature_encoder(input_feats)

        # Dropout layer
        dropout_layer = Dropout(self.dropout, name='dropout_block')

        encoder_output = dropout_layer(encoder_features)

        # Autoregressive model
        autoregressive_model = GRU(self.gru_units, return_sequences=True, name='autoregressive_layer')
        autoregressive_output = autoregressive_model(encoder_output)

        if self.input_attention:
            # add self-attention layer after input features
            attention_output, _ = AttentionWeights(name='attention_layer', causal=True)([autoregressive_output,
                                                                                         autoregressive_output])
            autoregressive_output = attention_output

        autoregressive_output = dropout_layer(autoregressive_output)

        # Contrastive loss
        contrastive_loss = ContrastiveLoss(self.gru_units, self.neg, self.steps)
        contrastive_loss_output = contrastive_loss([encoder_features, autoregressive_output])

        # Model
        self.model = Model(input_feats, contrastive_loss_output)

        print(self.model.summary())

    def train(self):
        """
        Train a CPC model
        :return: a trained model saved on disk
        """
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss={'Contrastive_Loss': lambda y_true, y_pred: y_pred})

        callbacks = super(CPCModel, self).train()

        x_val, _ = load_training_file(self.path_validation_data, shift=False)
        # Create dummy prediction so that Keras does not raise an error for wrong dimension
        y_val = np.random.rand(x_val.shape[0], 1, 1)
        # Check input data schedule
        if self.data_schedule == 'all':
            # Train the model
            x_train, _ = load_all_training_data(self.path_train_data, shift=False)
            y_train = np.random.rand(x_train.shape[0], 1, 1)
            self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                           validation_data=(x_val, y_val), callbacks=callbacks)
        elif self.data_schedule == 'epoch':
            path_input_data = load_epoch_training_data(self.path_train_data)
            for idx, path_data in enumerate(path_input_data):
                x_train, _ = load_training_file(path_data, shift=False)
                y_train = np.random.rand(x_train.shape[0], 1, 1)
                self.model.fit(x_train, y_train, epochs=idx + 1, batch_size=self.batch_size,
                               validation_data=(x_val, y_val), initial_epoch=idx, callbacks=callbacks)

        return self.model
