#!/usr/bin/env python3

from Encoder import Encoder
from Decoder import Decoder

import tensorflow as tf
import numpy as np
import Preprocessing as pre
import time
import os


class Hypothesis:
    def __init__(self, logit):
        self.word_ids = []
        self.value = np.double(logit)

    def add_hidden(self, h):
        self.hidden = h

    def add_word_id(self, word_id):
        self.word_ids.append(word_id)

    def add_dec_out(self, out):
        self.dec_out = out

    def set_history(self, path):
        self.word_ids.extend(path.word_ids)

    def add_dec_input(self, dec_input):
        self.dec_input = dec_input

    def update_value(self, logit):
        self.logits += logit

    def add_word(self, word):
        self.value += " " + word


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=real, logits=pred) * mask
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

        titulky, komentare = pre.get_data_lists(args)

        titulky= list(map(lambda x: pre.limit_seq_length(x,args.max_length),titulky))
        komentare= list(map(lambda x: pre.limit_seq_length(x,args.max_length),komentare))
        # print(list(titulky))


        # create vocab and tokenize

        self.tokenizer.fit_on_texts(titulky)
        self.tokenizer.fit_on_texts(komentare)
        #
        # with open("")

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

    def train_batch(self, inp, targ, enc_hidden, batch_size):

        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self._model.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            #
            # print(self.tokenizer.word_index)
            dec_input = tf.expand_dims(
                [self.tokenizer.word_index['<sos>']] * self.batch_size, 1)
            # forcing
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self._model.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = loss / batch_size
            variables = self._model.encoder.variables + \
                self._model.decoder.variables
            gradients = tape.gradient(batch_loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train_epoch(self, args, dataset):
        with open("loss_overfit","a") as loss_log:
            steps_per_epoch = args.max_sentences // args.batch_size
            enc_hidden = self._model.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_batch(
                    inp, targ, enc_hidden, args.batch_size)
                # print(inp.shape)
                total_loss += batch_loss
                loss_log.write(str(batch_loss.numpy()))
                loss_log.write("\n")
                print('Batch {} Loss {:.4f}'.format(batch, batch_loss.numpy()))

            return total_loss

    def train(self, args, dataset):

        check_dir = './training_checkpoints'
        check_prefix = os.path.join(check_dir, "overfit")
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self._model)
        for epoch in range(args.epochs):
            start = time.time()
            epoch_loss = self.train_epoch(args, dataset)
            print('Epoch {} had {} LOSS finished in {} seconds.'.format(
                epoch, epoch_loss, time.time() - start))

            if epoch % 5 == 4:
                checkpoint.save(file_prefix=check_prefix)


# force  the output to be of a certain length
    def greedy_evaluate(self, args, sentence):

        ids = []
        sentence = pre.preprocess(sentence,eval=True)
        tensor = self.tokenizer.texts_to_sequences([sentence])
        result = []
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
                result.append(predicted_word)
                dec_input = tf.expand_dims([predicted_id], 0)
            else:
                return " ".join(result), sentence, ids
        return " ".join(result), sentence, ids


# setting beam_size=1 should be equal to using greedy_evaluate
#super naive ATM


# return a sorted list of most probable outputs

    def beam_evaluate(self, args, sentence, beam_size=5):
        sentence = pre.preprocess(sentence,eval=True)
        seq = self.tokenizer.texts_to_sequences([sentence])

        hidden = tf.zeros((1, args.rnn_dim))
        # print()
        enc_out, enc_hidden = self._model.encoder(
            tf.convert_to_tensor(seq), hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self.tokenizer.word_index["<sos>"]], 0)

        predictions, dec_hidden, _ = self._model.decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
# val ind are (1,beam_size) tensors

        search = []
        values, indices = tf.math.top_k(predictions, beam_size)

        # constructs beggining of the search
        for i in range(beam_size):
            # neg log
            search.append(Hypothesis(-tf.math.log(values[0][i])))

            search[-1].add_word_id(indices[0][i].numpy())
            search[-1].add_hidden(dec_hidden)
            # need to expand dims input to decoder has to be a tensor of 1,1,dim
            search[-1].add_dec_input(tf.expand_dims([indices[0][i]],0))

# until some stopping condition is met
        end_condition=False

        while not end_condition:
            new_paths = []
            for path in search:
                predictions, dec_hidden, _ = self._model.decoder(path.dec_input,
                                                                 path.hidden,
                                                                 enc_out)
                values, indices = tf.math.top_k(predictions, beam_size)

                for i in range(beam_size):

                    # zabalit do nejakeho update
                    new_paths.append(Hypothesis(-tf.math.log(values[0][i]) +
                                                path.value))
                    new_paths[-1].set_history(path)
                    new_paths[-1].add_word_id(indices[0][i].numpy())
                    new_paths[-1].add_hidden(dec_hidden)
                    new_paths[-1].add_dec_input(tf.expand_dims([indices[0][i]],0))

            # keep only the best paths
            search = sorted(new_paths, key=lambda x: x.value)[:beam_size]

            # end condition checks
            for x in search:
                if self.tokenizer.index_word[x.word_ids[-1]]=="<eos>":
                    end_condition=True
                    break
                # limiting the length of decoded sentence
                elif len(x.word_ids) > 10:
                    end_condition=True
                    break

        res=[]
        for path in search:
            x = [ self.tokenizer.index_word[id]  for id in path.word_ids ]
            if x[-1]=="<eos>":
                del x[-1]
            res.append(" ".join(x))
# just returning dummy values for unpacking
        return res , None , None
