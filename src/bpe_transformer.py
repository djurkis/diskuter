#!/usr/bin/env python3

import re
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from Preprocessing import *
import matplotlib.pyplot as plt


tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)


def clean(sent):
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.replace("quot", "")
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    return w


def get_generator_bytes(maximum=10000):
    """Returns generators of headlines and commentares"""
    titulky = []
    komentare = []
    with open("/home/jurkis/diskuter/data/half_filtered", 'rb') as h:
        data = pickle.load(h)
        data = data[:maximum]

        for clanok in data:
            for koment in clanok:
                titulky.append(
                    clean(koment["title"]).encode())
                komentare.append(
                    clean(koment["reaction"]).encode())
                break

    return (x for x in titulky), (x for x in komentare)


BUFFER_SIZE = 20000
BATCH_SIZE = 32

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**9)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**9)


def encode(l1, l2):
    l1 = [tokenizer_en.vocab_size] + \
        tokenizer_en.encode(l1.numpy()) + [tokenizer_en.vocab_size + 1]

    l2 = [tokenizer_pt.vocab_size] + \
        tokenizer_pt.encode(l1.numpy()) + [tokenizer_pt.vocab_size + 1]

    return l1, l2


def filter_max_length(x, y, max=10):
    return tf.logical_and(tf.size(x) <= max, tf.size(y) <= max)


def tf_encode(source, target):
    return tf.py_function(encode, [source, target], [tf.int64, tf.int64])


train_dataset = train_examples.map(tf_encode)
# print(type(train_dataset))
# raise SystemExit(0)

train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# pos_encoding = positional_encoding(50, 512)
# print (pos_encoding.shape)
#
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model % self.num_heads == 0

            self.depth = d_model // self.num_heads

            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

            self.dense = tf.keras.layers.Dense(d_model)

        def split_heads(self, x, batch_size):
            """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, v, k, q, mask):
            batch_size = tf.shape(q)[0]

            q = self.wq(q)  # (batch_size, seq_len, d_model)
            k = self.wk(k)  # (batch_size, seq_len, d_model)
            v = self.wv(v)  # (batch_size, seq_len, d_model)

            # (batch_size, num_heads, seq_len_q, depth)
            q = self.split_heads(q, batch_size)
            # (batch_size, num_heads, seq_len_k, depth)
            k = self.split_heads(k, batch_size)
            # (batch_size, num_heads, seq_len_v, depth)
            v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)

            # (batch_size, seq_len_q, num_heads, depth)
            scaled_attention = tf.transpose(
                scaled_attention, perm=[0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention,
                                          (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

            # (batch_size, seq_len_q, d_model)
            output = self.dense(concat_attention)

            return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
