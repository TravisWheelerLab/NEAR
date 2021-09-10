import math
import os
import tensorflow as tf

sess = tf.compat.v1.Session()


def _set_padding_to_sentinel(padded_representations, sequence_lengths,
                             sentinel):
    """Set padding on batch of padded representations to a sentinel value.

    Useful for preparing a batch of sequence representations for max or average
    pooling.

    Args:
      padded_representations: float32 tensor, shape (batch, longest_sequence, d),
        where d is some arbitrary embedding dimension. E.g. the output of
        tf.data.padded_batch.
      sequence_lengths: tensor, shape (batch,). Each entry corresponds to the
        original length of the sequence (before padding) of that sequence within
        the batch.
      sentinel: float32 tensor, shape: broadcastable to padded_representations.

    Returns:
      tensor of same shape as padded_representations, where all entries
        in the sequence dimension that came from padding (i.e. are beyond index
        sequence_length[i]) are set to sentinel.
    """
    sequence_dimension = 1
    embedding_dimension = 2

    with tf.variable_scope('set_padding_to_sentinel', reuse=False):
        longest_sequence_length = tf.shape(
            padded_representations)[sequence_dimension]
        embedding_size = tf.shape(padded_representations)[embedding_dimension]

        seq_mask = tf.sequence_mask(sequence_lengths, longest_sequence_length)
        seq_mask = tf.expand_dims(seq_mask, [embedding_dimension])
        is_not_padding = tf.tile(seq_mask, [1, 1, embedding_size])

        full_sentinel = tf.zeros_like(padded_representations)
        full_sentinel = full_sentinel + tf.convert_to_tensor(sentinel)

        per_location_representations = tf.where(
            is_not_padding, padded_representations, full_sentinel)

        return per_location_representations


def _residual_block(sequence_features, sequence_lengths, hparams, layer_index,
                    activation_fn, is_training):
    """Construct a single block for a residual network."""

    with tf.variable_scope('residual_block_{}'.format(layer_index), reuse=False):
        shifted_layer_index = layer_index - hparams.first_dilated_layer + 1
        dilation_rate = max(1, hparams.dilation_rate ** shifted_layer_index)

        num_bottleneck_units = math.floor(
            hparams.resnet_bottleneck_factor * hparams.filters)

        features = _batch_norm(sequence_features, is_training)
        features = activation_fn(features)
        features = _conv_layer(
            sequence_features=features,
            sequence_lengths=sequence_lengths,
            num_units=num_bottleneck_units,
            dilation_rate=dilation_rate,
            kernel_size=hparams.kernel_size,
        )
        features = _batch_norm(features, is_training=is_training)
        features = activation_fn(features)

        # The second convolution is purely local linear transformation across
        # feature channels, as is done in
        # third_party/tensorflow_models/slim/nets/resnet_v2.bottleneck
        residual = _conv_layer(
            features,
            sequence_lengths,
            num_units=hparams.filters,
            dilation_rate=1,
            kernel_size=1)

        with_skip_connection = sequence_features + residual
        return with_skip_connection


def _conv_layer(sequence_features, sequence_lengths, num_units, dilation_rate,
                kernel_size):
    """Return a convolution of the input features that respects sequence len."""
    padding_zeroed = _set_padding_to_sentinel(sequence_features, sequence_lengths,
                                              tf.constant(0.))
    conved = tf.layers.conv1d(
        padding_zeroed,
        filters=num_units,
        kernel_size=[kernel_size],
        dilation_rate=dilation_rate,
        padding='same')

    # Re-zero padding, because shorter sequences will have their padding
    # affected by half the width of the convolution kernel size.
    re_zeroed = _set_padding_to_sentinel(conved, sequence_lengths,
                                         tf.constant(0.))
    return re_zeroed


def _make_representation(features, hparams, mode):
    """Produces [batch_size, sequence_length, embedding_dim] features.

  Args:
    features: dict from str to Tensor, containing sequence and sequence length.
    hparams: tf.contrib.training.HParams()
    mode: tf.estimator.ModeKeys instance.

  Returns:
    Tensor of shape [batch_size, sequence_length, embedding_dim].
  """
    sequence_features = features[protein_dataset.SEQUENCE_KEY]
    sequence_lengths = features[protein_dataset.SEQUENCE_LENGTH_KEY]

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # initial embedding shape
    sequence_features = _conv_layer(
        sequence_features=sequence_features,
        sequence_lengths=sequence_lengths,
        num_units=hparams.filters,
        dilation_rate=1,
        kernel_size=hparams.kernel_size,
    )

    for layer_index in range(hparams.num_layers):
        sequence_features = _residual_block(
            sequence_features=sequence_features,
            sequence_lengths=sequence_lengths,
            hparams=hparams,
            layer_index=layer_index,
            activation_fn=tf.nn.relu,
            is_training=is_training)

    return sequence_features

def _batch_norm(features, is_training):
    return tf.layers.batch_normalization(features, training=is_training)


if __name__ == '__main__':
    pth = '/home/tc229954/data/prefilter/proteinfer/trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760/'
    model = tf.saved_model.load(sess, ['serve'], pth)
    from pprint import pprint
    pprint(tf.trainable_variables())
    exit()
    for n in tf.get_default_graph().get_operations():
        print(n)
        break
