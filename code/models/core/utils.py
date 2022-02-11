"""
@date 08.02.2022
Objects needed for loading a Contrastive Predictive Coding model ["Representation Learning with Contrastive
Predictive Coding", van den Oord et al., 2018]
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Dense, Conv1D, Layer, Conv2DTranspose, Lambda
from tensorflow.python.keras.layers import Attention
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

__docformat__ = ['reStructuredText']
__all__ = ['FeatureEncoder', 'ContrastiveLoss']


class Block(Layer):
    """
    Super class for all the blocks so they have get_layer method. The method is used in prediction to extract either
    features of the APC encoder or the CPC encoder
    """

    def __init__(self, name):
        super(Block, self).__init__(name=name)

    def get_layer(self, name=None, index=None):
        """
        Keras sourcecode for Model.
        :param name: String name of the layer
        :param index: int index of the layer
        :return: the layer if name or index is found, error otherwise
        """
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                                 ' but model only has ' + str(len(self.layers)) +
                                 ' layers.')
            else:
                return self.layers[index]
        else:
            if not name:
                raise ValueError('Provide either a layer name or layer index.')
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError('No such layer: ' + name)


class ContrastiveLoss(Block):
    """
    It creates the block that calculates the contrastive loss for given latent representation and context
    representations. Implementation from wav2vec
    (https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec.py)
    [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)
    """

    def __init__(self, context_units, neg, steps, name='Contrastive_Loss'):
        """
        :param context_units: Number of units of the context representation
        :param neg: Number of negative samples
        :param steps: Number of steps to predict
        :param name: Name of the block, by default Contrastive_Loss
        """
        super(ContrastiveLoss, self).__init__(name=name)
        self.neg = neg
        self.steps = steps
        self.context_units = context_units
        self.layers = []
        with K.name_scope(name):
            self.project_steps = Conv2DTranspose(self.steps, kernel_size=1, strides=1, name='project_steps')
            self.project_latent = Conv1D(self.context_units, kernel_size=1, strides=1, name='project_latent')
            self.expand_dim_ctxt = Lambda(lambda x: K.expand_dims(x, -1))
            self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                         reduction=tf.keras.losses.Reduction.SUM)
            self.layers.append(self.project_steps)
            self.layers.append(self.project_latent)

    def get_negative_samples(self, true_features):
        """
        It calculates the negative samples re-ordering the time-steps of the true features.
        :param true_features: A tensor with the apc predictions for the input.
        :return: A tensor with the negative samples.
        """
        # Shape SxTxF
        samples = K.shape(true_features)[0]
        timesteps = K.shape(true_features)[1]
        features = K.shape(true_features)[2]

        # New shape FxSxT
        true_features = K.permute_dimensions(true_features, pattern=(2, 0, 1))
        # New shape Fx (S*T)
        true_features = K.reshape(true_features, (features, -1))

        high = timesteps

        # New order for time-steps
        indices = tf.repeat(tf.expand_dims(tf.range(timesteps), axis=-1), self.neg)
        neg_indices = tf.random.uniform(shape=(samples, self.neg * timesteps), minval=0, maxval=high - 1,
                                        dtype=tf.dtypes.int32)
        neg_indices = tf.where(tf.greater_equal(neg_indices, indices), neg_indices + 1, neg_indices)

        right_indices = tf.reshape(tf.range(samples), (-1, 1)) * high
        neg_indices = neg_indices + right_indices

        # Reorder for negative samples
        negative_samples = tf.gather(true_features, tf.reshape(neg_indices, [-1]), axis=1)
        negative_samples = K.permute_dimensions(K.reshape(negative_samples,
                                                          (features, samples, self.neg, timesteps)),
                                                (2, 1, 3, 0))
        return negative_samples

    def call(self, inputs, **kwargs):
        """
        :param inputs: A list with two elements, the latent representation and the context representation
        :param kwargs:
        :return: the contrastive loss calculated
        """
        true_latent, context_latent = inputs

        # Linear transformation of latent representation into the vector space of context representations
        true_latent = self.project_latent(true_latent)

        # Calculate the following steps using context_latent
        context_latent = self.expand_dim_ctxt(context_latent)
        # context_latent = K.expand_dims(context_latent, -1)
        predictions = self.project_steps(context_latent)

        negative_samples = self.get_negative_samples(true_latent)

        true_latent = K.expand_dims(true_latent, 0)

        targets = K.concatenate([true_latent, negative_samples], 0)
        copies = self.neg + 1  # total of samples in targets

        # samples, timesteps, features, steps = predictions.shape

        # Logits calculated from predictions and targets
        logits = None

        for i in range(self.steps):
            if i == 0:
                # The time-steps are corresponding as is the first step.
                logits = tf.reshape(tf.einsum("stf,cstf->tsc", predictions[:, :, :, i], targets[:, :, :, :]), [-1])
            else:
                # We need to match the time-step taking into account the step for which is being predicted
                logits = tf.concat([logits, tf.reshape(tf.einsum("stf,cstf->tsc", predictions[:, :-i, :, i],
                                                                 targets[:, :, i:, :]), [-1])], 0)

        logits = tf.reshape(logits, (-1, copies))
        total_points = tf.shape(logits)[0]

        # Labels, this should be the true value, that is 1.0 for the first copy (positive sample) and 0.0 for the
        # rest.
        label_idx = [True] + [False] * self.neg
        labels = tf.where(label_idx, tf.ones((total_points, copies)), tf.zeros((total_points, copies)))

        # The loss is the softmax_cross_entropy_with_logits sum over copies (classes, true and negs) and mean for all
        # steps and samples
        loss = self.cross_entropy(labels, logits)
        loss = tf.reshape(loss, (1,))
        return loss

    def get_config(self):
        return {'context_units': self.context_units, 'neg': self.neg, 'steps': self.steps}


class FeatureEncoder(Block):
    """
    It creates a keras layer for the encoder part (latent representations)
    """

    def __init__(self, n_layers, units, dropout, name='Feature_Encoder'):
        """
        :param n_layers: Number of convolutional layers
        :param units: Number of filters per convolutional layer
        :param dropout: Percentage of dropout between layers
        :param name: Name of the block, by default Feature_Encoder
        """
        super(FeatureEncoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.layers = []
        with K.name_scope(name):
            for i in range(n_layers):
                self.layers.append(Dense(units, activation='relu', name='dense_layer_' + str(i)))
                if i == n_layers - 1:
                    self.layers.append(Dropout(dropout, name='cpc_latent_layer'))
                else:
                    self.layers.append(Dropout(dropout, name='dense_dropout_' + str(i)))

    def call(self, inputs, **kwargs):
        """
        It is execute when an input tensor is passed
        :param inputs: A tensor with the input features
        :return: A tensor with the output of the block
        """
        features = inputs
        for layer in self.layers:
            features = layer(features)
        return features

    def get_config(self):
        return {'n_layers': self.n_layers, 'units': self.units, 'dropout': self.dropout}


class AttentionWeights(Attention):

    def _apply_scores(self, scores, value, scores_mask=None):
        """
        Overwrites the base method to output the weights as well
        :param scores: scores tensor
        :param value: value tensor
        :param scores_mask: mask to apply to scores
        :return: attention calculation and weights (two tensors)
        """
        if scores_mask is not None:
            padding_mask = math_ops.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention distribution.
            scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
        weights = nn.softmax(scores)
        return math_ops.matmul(weights, value), weights


class ModelCheckpointBatch(tf.keras.callbacks.ModelCheckpoint):
    """
    Allows to have batch number in the checkpoint file path.
    This is not required in newer tf versions (https://github.com/tensorflow/tensorflow/pull/38669/files)
    """
    def __init__(self, filepath, **kwargs):
        """
        Add total samples seen in epoch
        """
        super(ModelCheckpointBatch, self).__init__(filepath, **kwargs)
        self._samples_seen_in_epoch = 0

    def on_batch_end(self, batch, logs=None):
        """
        Overwrites the base method so it uses batch number in the file path.
        :param batch: number of batch
        :param logs: logs (e.g., validation loss, number of seen samples)
        """
        logs = logs or {}
        if isinstance(self.save_freq, int):
            self._samples_seen_since_last_saving += logs.get('size', 1)
            self._samples_seen_in_epoch += self._samples_seen_since_last_saving
            if self._samples_seen_since_last_saving >= self.save_freq:
                filepath = self.filepath.format(epoch=self._current_epoch+1, batch=batch, **logs)
                self.model.save(filepath)
                if self.verbose > 0:
                    print(f'\nEpoch {self._current_epoc + 1:d} - Samples: {self._samples_seen_in_epoch} saving model '
                          f'to {filepath}')
                self._samples_seen_since_last_saving = 0
