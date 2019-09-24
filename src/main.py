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
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=20000,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=512, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--threads", default=8, type=int,
                        help="Maximum number of threads to use.")
    parser.add_argument("--vocab_limit", default=80000, type=int,
                        help="Maximum number of words to use in vocab.")
    parser.add_argument("--lr", default=0.002, type=float,
                        help="Learning rate for optimizer.")
    parser.add_argument("--max_length", default=8, type=int,
                        help="Maximum length of output sentence., 0  for no limit")
    args = parser.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))


    # tokenize and check how many words
    network = Network(args, args.vocab_limit)

    data = network.prepare_data(args)
    network.train(args, data)



# test evalutaion
    titulky,komentare = pre.get_data_lists(args)

    with open("gold_labels_overfit2","w") as gold, open("output_overfit2","w") as out,open("inputs_overfit2","w") as inputs:
        for i, (t, k) in enumerate(zip(titulky, komentare)):

            result, sent, ids = network.beam_evaluate(args, " ".join(t.split()[:args.max_length]))

            inputs.write("{}".format( " ".join(t.split()[:args.max_length])))
            inputs.write("\n")
            out.write("{}".format(result))
            out.write("\n")
            gold.write("{}".format(" ".join(k.split()[:args.max_length])))
            gold.write("\n")
