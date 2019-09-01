#!/usr/bin/env python3

import re
import pickle
import tensorflow_datasets as tfds


# splitting is wasting time, just append and prepend tokens
def preprocess(sent):
    """returns a cleaned sentence with eos and sos tokens"""
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.replace("quot", "")
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    x = w.split(' ')
    w = '<SOS> ' + " ".join(x) + ' <EOS>'
    return w


def limit_sent_length(sent, max):
    a = sent.split()
    if len(a) - 2 < max:
        return sent
    else:
        return '<SOS> ' + " ".join(a[1:max + 1]) + ' <EOS>'


# hardcoded path to file....
def get_data(args, max=0):
    """Returns headlines and commentaries paired by indexes"""
    titulky = []
    komentare = []
    with open("/home/jurkis/diskuter/data/half_filtered", 'rb') as h:
        data = pickle.load(h)
        maximum = args.max_sentences if max == 0 else max
        data = data[:maximum]

        for clanok in data:
            for koment in clanok:
                titulky.append(limit_sent_length(
                    preprocess(koment["title"]), 10))
                komentare.append(limit_sent_length(
                    preprocess(koment["reaction"]), 10))
                break
    return titulky, komentare




if __name__ == "__main__":
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    tit,kom = get_generators()
