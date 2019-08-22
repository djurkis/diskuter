#!/usr/bin/env python3

import re
import pickle

def preprocess(sent):
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.replace("quot", "")
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    x = w.split(' ')
    w = '<SOS> ' + " ".join(x) + ' <EOS>'
    return w


def limit_sent_length(sent,max):
    a = sent.split()
    if len(a) - 2 < max:
        return sent
    else:
        return '<SOS> ' + " ".join(a[1:max+1]) + ' <EOS>'


# hardcoded path to file....
def get_data(args,max=0):
    """Returns headlines and commentaries paired by indexes"""

    titulky = []
    komentare = []
    with open("/home/jurkis/diskuter/data/half_filtered", 'rb') as h:
        data = pickle.load(h)
        maximum = args.max_sentences if max==0 else max
        data = data[:maximum]
        
        for clanok in data:
            for koment in clanok:
                titulky.append(preprocess(koment["title"]))
                komentare.append(preprocess(koment["reaction"]))
                break
    return titulky , komentare
