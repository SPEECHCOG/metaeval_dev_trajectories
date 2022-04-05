"""
@date 08.02.2022
Contrastive Predictive Coding model
[Representation Learning with Contrastive Predictive Coding]
"""

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from core.apc import AttentionWeights
from core.model_base import ModelBase
from core.utils import FeatureEncoder, ContrastiveLoss

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

        # training
        super(CPCModel, self).train()

        return self.model
