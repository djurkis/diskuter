#!/usr/bin/env python3
import numpy as np
import pickle
import re
import os
import tensorflow as tf
import sys
import time


def preprocess(sent, maximum_length=40):
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    w = '<SOS> ' + w[:maximum_length] + ' <EOS>'
    return w


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Network:
    def __init__(self, args, num_source_chars):
        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()

                self.encoder = Encoder(
                    num_source_chars, args.embed_dim, args.rnn_dim, args.batch_size)
                self.decoder = Decoder(
                    num_source_chars, args.embed_dim, args.rnn_dim, args.batch_size)
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=args.lr)

                self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction='none')
        self._model = Model()
# todo
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=args.vocab_limit, oov_token="<unk>")
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # self.loss_function=

        # writers

    @tf.function
    def lossf(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # returns a tf.dataset object that is batchable

    def prepare_data(self, args, data):

        # take some data and divide
        print("tokens")
        data = data[:args.max_sentences]
        titulky = []
        komentare = []
        for clanok in data:
            for koment in clanok:
                titulky.append(preprocess(koment["title"]))
                komentare.append(preprocess(koment["reaction"]))

        # create vocab and tokenize
        self.tokenizer.fit_on_texts(titulky)
        self.tokenizer.fit_on_texts(komentare)
        tensor_titulky = self.tokenizer.texts_to_sequences(titulky)
        tensor_koment = self.tokenizer.texts_to_sequences(komentare)
        # padding
        input_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(
            tensor_titulky, padding="post")
        target_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(
            tensor_koment, padding="post")

        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)).shuffle(len(input_tensor_train))

        dataset = dataset.batch(args.batch_size, drop_remainder=True)
        return dataset


    def train_batch(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self._model.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [self.tokenizer.word_index['<sos>']] * args.batch_size, 1)
            # forcing
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self._model.decoder(
                    dec_input, dec_hidden, enc_output)

                loss += self.lossf(targ[:, t], predictions)
                # print(predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)


            batch_loss = (loss / int(targ.shape[1]))

            variables = self._model.encoder.trainable_variables + \
                self._model.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    def train_epoch(self, args, dataset):
        # zbytocne
        steps_per_epoch = args.max_sentences // args.batch_size

        enc_hidden = self._model.encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = self.train_batch(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Batch {} Loss {:.4f}'.format(batch, batch_loss.numpy()))


    def train(self, args, dataset):
        for epoch in range(args.epochs):
            start = time.time()
            self.train_epoch(args, dataset)
            print('Epoch {} finished in {} seconds.'.format(
                epoch, time.time() - start))


if __name__ == "__main__":
    import argparse
    import datetime

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch size.")
    parser.add_argument("--embed_dim", default=32, type=int,
                        help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=2000,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=64, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--vocab_limit", default=30000, type=int,
                        help="Maximum number of words to use in vocab.")
    parser.add_argument("--lr", default=0.001, type=float,
                        help="Learning rate for optimizer.")
    args = parser.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))


# dumb arguments
    network = Network(args, args.vocab_limit)

    with open("half_filtered", 'rb') as h:
        raw_data = pickle.load(h)

    print("asd")
    data = network.prepare_data(args, raw_data)
    network.train(args, data)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    # out_path="lemmatizer_competition_test.txt"
    # if os.path.isdir(args.logdir):
    #     out_path=os.path.join(args.logdir, out_path)
    # with open(out_path, "w", encoding = "utf-8") as out_file:
    #     for i, sentence in enumerate(network.predict(morpho.test, args)):
    #
    #         for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
    #             lemma=[]
    #             for c in map(int, sentence[j]):
    #                 if c == MorphoDataset.Factor.EOW:
    #                     break
    #                 lemma.append(
    #                     morpho.test.data[morpho.test.LEMMAS].alphabet[c])
    #
    #             print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
    #                   "".join(lemma),
    #                   morpho.test.data[morpho.test.TAGS].word_strings[i][j],
    #                   sep = "\t", file = out_file)
    #         print(file = out_file)
