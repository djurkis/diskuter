#!/usr/bin/env python3
from Network import Network
import argparse
import datetime
import numpy as np
import tensorflow as tf
import Preprocessing as pre

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch size.")
    parser.add_argument("--embed_dim", default=128, type=int,
                        help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=20,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=256, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--vocab_limit", default=10000, type=int,
                        help="Maximum number of words to use in vocab.")
    parser.add_argument("--lr", default=0.003, type=float,
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

    data = network.prepare_data(args)
    network.train(args, data)



# test evalutaion
    titulky,komentare = pre.get_data(args,max=5)

    for i, (t, k) in enumerate(zip(titulky, komentare)):
        if i > 10:
            break
        result, sent, ids = network.beam_evaluate(args, t)
        print("{} -- {}".format(result,k))
