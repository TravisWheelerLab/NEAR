import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def super_duper_kmer(maxlen, n_classes, embed_dim=32, vocab_size=23):

    inputs = layers.Input(shape=(None, 23, 1))
    # embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    # x = embedding_layer(inputs)
    gap = layers.GlobalAveragePooling2D()
    act = layers.ReLU()
    acts = []

    for ks in [8, 12, 16, 20, 24, 28, 32, 36]:

        x = layers.Conv2D(64, (ks, 1), padding='same')(inputs)
        #x = layers.BatchNormalization()(x)
        x = act(x)
        # x = layers.Conv2D(64, ks, padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = act(x)
        gapped = gap(x)
        acts.append(gapped)

    x = tf.concat(acts, axis=-1)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=x)


def attn_model(maxlen, n_classes, embed_dim=32, num_heads=6, ff_dim=512,
        vocab_size=23):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
