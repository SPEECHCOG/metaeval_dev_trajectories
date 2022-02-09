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
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Input, GRU, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from core.utils import FeatureEncoder, ContrastiveLoss
from core.model_base import ModelBase
from core.apc import AttentionWeights

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
        :param x_train: Input training data
        :param y_train: Output training data (not used in this model)
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

        # Input shape
        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

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
            # query = Dense(self.features, use_bias=False, name='query_attention')(encoder_features)
            # value = Dense(self.features, use_bias=False, name='value_attention')(encoder_features)
            attention_output, _ = AttentionWeights(name='attention_layer', causal=True)([autoregressive_output,
                                                                                         autoregressive_output])
            # input_feats_att = Concatenate()([autoregressive_output, attention_output])
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

        callbacks = []

        if self.statistical_analysis:
            model_full_path = os.path.join(self.full_path_output_folder,
                                           self.configuration['statistical_analysis']['system'],
                                           str(self.configuration['statistical_analysis']['model_id']))
            os.makedirs(model_full_path, exist_ok=True)
            model_file_name = os.path.join(self.full_path_output_folder,
                                           self.configuration['statistical_analysis']['system'],
                                           str(self.configuration['statistical_analysis']['model_id']),
                                           self.language +
                                           datetime.now().strftime("_%Y_%m_%d-%H_%M"))
            model_file_name_txt = os.path.join(self.full_path_output_folder,
                                               self.configuration['statistical_analysis']['system'],
                                               self.language +
                                               datetime.now().strftime("_%Y_%m_%d-%H_%M"))
        else:
            # Model file name for checkpoint and log
            model_file_name = os.path.join(self.full_path_output_folder, self.language +
                                           datetime.now().strftime("_%Y_%m_%d-%H_%M"))
            model_file_name_txt = model_file_name

        # log
        self.write_log(model_file_name_txt + '.txt')

        if self.save_untrained:
            folder_path, file_name = os.path.split(model_file_name_txt)
            self.model.save(f'{folder_path}/untrained_{file_name}.h5')

        # Callbacks for training
        # Adding early stop based on validation loss and saving best model for later prediction
        if self.statistical_analysis:
            checkpoint = ModelCheckpoint(model_file_name + '-{epoch:d}_{val_loss:.6f}' + '.h5', monitor='val_loss',
                                         mode='min', verbose=1,
                                         period=self.configuration['statistical_analysis']['period'])
            callbacks.append(checkpoint)
        else:
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
            # lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min')
            checkpoint = ModelCheckpoint(model_file_name + '.h5', monitor='val_loss', mode='min',
                                     verbose=1, save_best_only=True)
            callbacks.append(early_stop)
            callbacks.append(checkpoint)

        # Tensorboard
        log_dir = os.path.join(self.logs_folder_path, self.language, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)
        callbacks.append(tensorboard)

        # Train the model
        # Create dummy prediction so that Keras does not raise an error for wrong dimension
        y_dummy = np.random.rand(self.x_train.shape[0], 1, 1)

        if self.x_val is not None:
            y_val_dummy = np.random.rand(self.x_val.shape[0], 1, 1)
            self.model.fit(x=self.x_train, y=y_dummy, epochs=self.epochs, batch_size=self.batch_size,
                           validation_data=(self.x_val, y_val_dummy), callbacks=callbacks)
        else:
            self.model.fit(x=self.x_train, y=y_dummy, epochs=self.epochs, batch_size=self.batch_size,
                           validation_split=0.3, callbacks=callbacks)

        return self.model
