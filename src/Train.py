#!/usr/bin/env python3
import numpy as np
import pickle
import re
import os
import tensorflow as tf
import sys
import time


def preprocess(sent, maximum_words=3):
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.replace("quot", "")
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    x = w.split(' ')[:3]
    w = '<SOS> ' + " ".join(x) + ' <EOS>'
    return w


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_sz = batch_size
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
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) +
                                  self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)


        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

class Network:
    def __init__(self, args, num_source_chars):
        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()

                self.encoder = Encoder(
                    num_source_chars, args.embed_dim, args.rnn_dim, args.batch_size)
                self.decoder = Decoder(
                    num_source_chars, args.embed_dim, args.rnn_dim, args.batch_size)

        self._model = Model()
# todo
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', num_words=args.vocab_limit, oov_token="<unk>")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)



    # returns a tf.dataset object that is batchable
    def prepare_data(self, args, data):

        # take some data and divide

        data = data[:args.max_sentences]
        titulky = []
        komentare = []
        for clanok in data:
            for koment in clanok:
                titulky.append(preprocess(koment["title"]))
                komentare.append(preprocess(koment["reaction"]))
                break

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

                # sem dat prerobeny loss
                loss += loss_function(targ[:,t], predictions)

                dec_input = tf.expand_dims(targ[:, t], 1)


            batch_loss = (loss / int(targ.shape[1]))
            variables = self._model.encoder.variables + \
                self._model.decoder.variables

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train_epoch(self, args, dataset):
        # zbytocne
        steps_per_epoch = args.max_sentences // args.batch_size

        enc_hidden = self._model.encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            # print(inp.shape)
            batch_loss = self.train_batch(inp, targ, enc_hidden)
            total_loss += batch_loss
            print(batch_loss)
            print('Batch {}/{} Loss {:.4f}'.format(batch,
                                                   steps_per_epoch, batch_loss.numpy()))

    def train(self, args, dataset):
        for epoch in range(args.epochs):
            start = time.time()
            self.train_epoch(args, dataset)
            print('Epoch {} finished in {} seconds.'.format(
                epoch, time.time() - start))

    def evaluate(self, args, sentence):
        ids = []
        sentence = preprocess(sentence)
        tensor = self.tokenizer.texts_to_sequences([sentence])
        result = ''
        hidden = tf.zeros((1, args.rnn_dim))

        enc_out, enc_hidden = self._model.encoder(
            tf.convert_to_tensor(tensor), hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tokenizer.word_index['<sos>']], 0)

        for t in range(args.max_length):
            predictions, dec_hidden, _ = self._model.decoder(dec_input,
                                                             dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            ids.append(predicted_id)

            predicted_word = self.tokenizer.index_word[predicted_id]

            # pokial nieje eos tak pokracuj v predikcii
            # blbost ... po eos nevie co ma dat...

            if (predicted_word != "<eos>"):
                result += predicted_word + ' '
                dec_input = tf.expand_dims([predicted_id], 0)
            else:
                return result, sentence, ids
        return result, sentence, ids





if __name__ == "__main__":
    import argparse
    import datetime

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2,
                        type=int, help="Batch size.")
    parser.add_argument("--embed_dim", default=4, type=int,
                        help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=2,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=4, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--vocab_limit", default=10000, type=int,
                        help="Maximum number of words to use in vocab.")
    parser.add_argument("--lr", default=0.001, type=float,
                        help="Learning rate for optimizer.")
    parser.add_argument("--max_length", default=40, type=int,
                        help="Maximum length of output sentence.")
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))


# dumb arguments

    # tokenize and check how many words
    network = Network(args, args.vocab_limit)

    with open("half_filtered", 'rb') as h:
        raw_data = pickle.load(h)

    data = network.prepare_data(args, raw_data)
    network.train(args, data)




    # for evalutiaon
    data = raw_data[:2]
    titulky = []
    komentare = []
    for clanok in data:
        for koment in clanok:
            titulky.append(preprocess(koment["title"]))
            komentare.append(preprocess(koment["reaction"]))
            break

    for i, (t, k) in enumerate(zip(titulky, komentare)):
        if i > 10:
            break
        result, sent, ids = network.evaluate(args, t)
        print("t= {} --> r={}  | l ={} ".format(t, result, k))
