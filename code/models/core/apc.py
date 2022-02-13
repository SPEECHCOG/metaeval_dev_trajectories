""""
@date 08.02.2022
Autoregressive Predictive Coding model
[An unsupervised autoregressive model for speech representation learning]

This corresponds to a translation from the Pytorch implementation
(https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) to Keras implementation
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from core.model_base import ModelBase
from core.utils import AttentionWeights
from read_configuration import load_all_training_data, load_training_file, load_epoch_training_data

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class APCModel(ModelBase):

    def write_log(self, file_path: str):
        """
        Write the log of the configuration parameters used for training an APC model
        :param file_path: path where the file will be saved
        :return: a text logfile
        """
        super(APCModel, self).write_log(file_path)

    def load_training_configuration(self, config):
        """
        It loads configuration from dictionary, and instantiates the model architecture
        :param config: a dictionary with the configuration parameters for training
        :return: an instance will have the parameters from configuration and the model architecture
        """
        super(APCModel, self).load_training_configuration(config)

        # Model architecture: PreNet (stacked linear layers with ReLU activation and dropout) ->
        # APC (multi-layer GRU network) -> Postnet(Conv1D this is only used during training)

        # Load apc configuration parameters
        apc_config = config['model']['apc']
        self.prenet = apc_config['prenet']
        self.prenet_layers = apc_config['prenet_layers']
        self.prenet_dropout = apc_config['prenet_dropout']
        self.prenet_units = apc_config['prenet_units']
        self.rnn_layers = apc_config['rnn_layers']
        self.rnn_dropout = apc_config['rnn_dropout']
        self.rnn_units = apc_config['rnn_units']
        self.residual = apc_config['residual']
        self.learning_rate = apc_config['learning_rate']
        self.steps_shift = apc_config['steps_shift']

        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')

        rnn_input = input_feats

        if self.prenet:
            for i in range(self.prenet_layers):
                rnn_input = Dense(self.prenet_units, activation='relu', name='prenet_linear_' + str(i))(rnn_input)
                rnn_input = Dropout(self.prenet_dropout, name='prenet_dropout_'+str(i))(rnn_input)

        # RNN
        for i in range(self.rnn_layers):
            # TODO Padding for sequences is not yet implemented
            if i+1 < self.rnn_layers:
                # All GRU layers will have rnn_units units except last one
                rnn_output = GRU(self.rnn_units, return_sequences=True, name='rnn_layer_'+str(i))(rnn_input)
            else:
                # Last GRU layer will have latent_dimension units
                if self.residual and self.latent_dimension == self.rnn_units:
                    # The latent representation will be then the output of the residual connection.
                    rnn_output = GRU(self.latent_dimension, return_sequences=True, name='rnn_layer_'+str(i))(rnn_input)
                else:
                    rnn_output = GRU(self.latent_dimension, return_sequences=True, name='latent_layer')(rnn_input)

            if i+1 < self.rnn_layers:
                # Dropout to all layers except last layer
                rnn_output = Dropout(self.rnn_dropout, name='rnn_dropout_'+str(i))(rnn_output)

            if self.residual:
                # residual connection is applied to last layer if the latent dimension and rnn_units are the same,
                # otherwise is omitted. And to the first layer if the PreNet units and RNN units are the same,
                # otherwise is omitted also for first layer.
                residual_last = (i+1 == self.rnn_layers and self.latent_dimension == self.rnn_units)
                residual_first = (i == 0 and self.prenet_units == self.rnn_units)

                if (i+1 < self.rnn_layers and i != 0) or residual_first:
                    rnn_input = Add(name='rnn_residual_'+str(i))([rnn_input, rnn_output])

                # Update output for next layer (PostNet) if this is the last layer. This will also be the latent
                # representation.
                if residual_last:
                    rnn_output = Add(name='latent_layer')([rnn_input, rnn_output])

                # Update input for next layer
                if not residual_first and i == 0:
                    # Residual connection won't be applied but we need to update input value for next RNN layer
                    # to the output of RNN + dropout
                    rnn_input = rnn_output
            else:
                # Output of the dropout or RNN layer in the case of the last layer
                rnn_input = rnn_output

        if self.input_attention:
            # add self-attention layer after rnn output
            attention_output, _ = AttentionWeights(name='attention_layer', causal=True)([rnn_output, rnn_output])
            input_feats_att = Concatenate()([rnn_output, attention_output])

            # After PreNet
            rnn_output = input_feats_att

        # PostNet
        postnet_layer = Conv1D(self.features, kernel_size=1, padding='same', name='postnet_conv1d')(rnn_output)

        # APC Model
        self.model = Model(input_feats, postnet_layer)

    def train(self):
        """
        Train APC model, optimiser adam and loss L1 (mean absolute error)
        :return: an APC trained model. The model is also saved in the specified folder (output_path param in the
                 training configuration)
        """
        # Configuration of learning process
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss='mean_absolute_error')
        # configurations of model
        callbacks, callbacks_first_epoch = super(APCModel, self).train()

        x_val, y_val = load_training_file(self.path_validation_data, shift=True, steps=self.steps_shift)

        # Check input data schedule
        if self.data_schedule == 'all':
            # Train the model
            x_train, y_train = load_all_training_data(self.path_train_data, shift=True, steps=self.steps_shift)
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
                    if epoch == 0 and self.monitor_first_epoch and idx == 0:
                        custom_callbacks = callbacks_first_epoch
                    else:
                        custom_callbacks = callbacks
                    x_train, y_train = load_training_file(path_data, shift=True, steps=self.steps_shift)
                    history = self.model.fit(x_train, y_train, epochs=epoch+1, batch_size=self.batch_size,
                                             validation_data=(x_val, y_val), initial_epoch=epoch,
                                             callbacks=custom_callbacks)
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

        return self.model
