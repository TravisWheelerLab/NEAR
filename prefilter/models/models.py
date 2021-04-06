import torch.nn as nn
import pytorch_lightning as pl

class AttnModel(pl.LightningModule):
     
    def __init__(self):
        super().__init__(self)



    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

def attn_model(maxlen, n_classes, multilabel, embed_dim=32, num_heads=2, ff_dim=256,
        vocab_size=23):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    act = layers.ReLU()
    acts = []

    x = embedding_layer(inputs)
    conv_filters = 32

    for ks in [8, 12, 16, 20, 24, 28, 32, 36]:
        # this needs to be optimized
        x = layers.Conv1D(conv_filters, kernel_size=(ks), padding='same')(x)
        x = act(x)
        x = layers.Conv1D(conv_filters, kernel_size=(ks), padding='same')(x) + x
        acts.append(x)

    trans = []
    embed_dim = conv_filters

    for act in acts:
        act = TransformerBlock(embed_dim, num_heads, ff_dim)(act)
        trans.append(act)

    x = tf.concat(trans, axis=-1)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = tf.reduce_mean(x, axis=-1) # convention to represent sequence information as 
    # the mean embedding vector... we'll see if this works.

    if multilabel:
        outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def DeepFam(n_classes: int, 
        embedding_dim: int,
        kernel_sizes: list,
        dropout_p: float,
        vocab_size: int,
        filters: int,
        pooling_layer_type: str,
        hidden_units: int,
        multilabel: bool
        ):

    inputs = layers.Input(shape=(None,))
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # TODO: get adaptive pooling
    max_pool = layers.GlobalAveragePooling1D()
    hidden = layers.Dense(hidden_units, activation='relu')
    batch_norm = layers.BatchNormalization(momentum=0.1)
    if multilabel:
        classification = layers.Dense(n_classes, activation='sigmoid')
    else:
        classification = layers.Dense(n_classes, activation='softmax')
    dropout = layers.Dropout(rate=dropout_p)
    act = layers.ReLU()

    embedded = embedding(inputs)

    activations = []
    for kernel in kernel_sizes:
        x = layers.Conv1D(filters,
                kernel_size=kernel,
                activation=None,
                kernel_initializer='glorot_uniform')(embedded)
        x = layers.BatchNormalization(momentum=0.1)(x)
        x = act(x)
        x = max_pool(x)
        activations.append(x)

    x = tf.concat(activations, axis=-1)

    x = dropout(x)
    x = hidden(x)
    x = layers.BatchNormalization(momentum=0.1)(x)
    x = act(x)
    x = classification(x)

    return keras.Model(inputs=inputs, outputs=x)

def make_deepfam(n_classes, multilabel):
    encoding_dim = 10
    kernel_sizes = [8, 12, 16, 20, 24, 28, 32, 36]
    n_filters = 150
    dropout = 0.3
    pooling_layer_type = 'max'
    vocab_size = 23
    n_filters  = 150
    n_hidden = 2000
    df = DeepFam(n_classes, encoding_dim, kernel_sizes, dropout, vocab_size,
            n_filters, 'none', n_hidden, multilabel)
    return df

def DeepNOG(n_classes: int, embedding_dim: int, kernel_sizes: list,
        dropout_p: float,
        vocab_size: int,
        filters: int,
        pooling_layer_type: str,
        classification_nodes: int,
        multilabel: bool
        ):

    inputs = layers.Input(shape=(None,))
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # TODO: get adaptive pooling
    max_pool = layers.GlobalAveragePooling1D()
    if multilabel:
        classification = layers.Dense(n_classes, activation='sigmoid')
    else:
        classification = layers.Dense(n_classes, activation='softmax')

    dropout = layers.Dropout(rate=dropout_p)

    embedded = embedding(inputs)

    activations = []

    for kernel in kernel_sizes:
        x = layers.Conv1D(filters,
                kernel_size=kernel,
                activation='selu',
                kernel_initializer='lecun_normal')(embedded)
        x = max_pool(x)
        activations.append(x)

    x = tf.concat(activations, axis=-1)
    x = dropout(x)
    x = classification(x)

    return keras.Model(inputs=inputs, outputs=x)


def make_deepnog(n_classes, multilabel):
    encoding_dim = 10
    kernel_sizes = [8, 12, 16, 20, 24, 28, 32, 36]
    n_filters = 150
    dropout = 0.3
    pooling_layer_type = 'max'
    vocab_size = 23
    n_filters  = 150
    n_hidden = 2000
    dn = DeepNOG(n_classes, encoding_dim, kernel_sizes, dropout, vocab_size,
            n_filters, 'none', n_hidden, multilabel)

    return dn



if __name__ == '__main__':

    model = make_deepfam(858)
    model.summary()
