#!/usr/bin/env python3

from Encoder import Encoder
from Decoder import Decoder

import tensorflow as tf
import numpy as np
import Preprocessing as pre
import time

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
        self.batch_size = args.batch_size
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', num_words=args.vocab_limit, oov_token="<unk>")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)


    def prepare_data(self, args):
        """tokenizes pads_sequences and returns the dataset"""

        titulky,komentare = pre.get_data(args)

        # create vocab and tokenize
        self.tokenizer.fit_on_texts(titulky)
        self.tokenizer.fit_on_texts(komentare)
        tensor_titulky = self.tokenizer.texts_to_sequences(titulky)
        tensor_koment = self.tokenizer.texts_to_sequences(komentare)
        # padding
        input_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(tensor_titulky, padding="post")
        target_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(tensor_koment, padding="post")

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
                [self.tokenizer.word_index['<sos>']] * self.batch_size, 1)
            # forcing

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self._model.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:,t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

            variables = self._model.encoder.variables + \
                self._model.decoder.variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def train_epoch(self, args, dataset):

        steps_per_epoch = args.max_sentences // args.batch_size
        enc_hidden = self._model.encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            # print(inp.shape)
            batch_loss = self.train_batch(inp, targ, enc_hidden)
            total_loss += batch_loss
            print('Batch {} Loss {:.4f}'.format(batch, batch_loss.numpy()))
            
        return total_loss


    def train(self, args, dataset):
        for epoch in range(args.epochs):
            start = time.time()
            epoch_loss = self.train_epoch(args, dataset)
            print('Epoch {} had {} LOSS finished in {} seconds.'.format(
                epoch,epoch_loss, time.time() - start))




# force  the output to be of a certain length
    def evaluate(self, args, sentence):

        ids = []
        sentence = pre.preprocess(sentence)
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

            if (predicted_word != "<eos>"):
                result += predicted_word + ' '
                dec_input = tf.expand_dims([predicted_id], 0)
            else:
                return result, sentence, ids
        return result, sentence, ids
