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



def clean(s):

        # vymazat 039 je hoax
        w = re.sub(r'[\s+]'," ",s)
        w = re.sub(r'[" "]+', " ", w)
        w = w.replace("&quot;", "")
        w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
        w = w.strip()
        return w

def get_test_data(args,):
    titulky = []
    komentare = []
    with open("/home/jurkis/diskuter/data/half_filtered", 'rb') as h:
        data = pickle.load(h)
        maximum = args.max_sentences if max == 0 else max
        data = data[15000:15500]

        for clanok in data:
            for koment in clanok:
                titulky.append(
                    preprocess(koment["title"]))
                komentare.append(
                    preprocess(koment["reaction"]))
                break
    return titulky, komentare

def textify_data():

    with open("/home/jurkis/diskuter/data/half_filtered", 'rb') as h:
        data = pickle.load(h)
        with open("input","w") as input, open("target","w") as target:
            for clanok in data:
                headline=clean(clanok[0]["title"])
                print(headline)
                for koment in clanok:
                    input.write(headline)
                    input.write('\n')
                    target.write(clean(koment["reaction"]))
                    target.write('\n')

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
                    preprocess(koment["title"]), 40))
                komentare.append(limit_sent_length(
                    preprocess(koment["reaction"]), 40))
                break
    return titulky, komentare


if __name__ == "__main__":
    pass
