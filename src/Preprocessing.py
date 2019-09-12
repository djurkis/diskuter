#!/usr/bin/env python3

import re
import pickle
import tensorflow_datasets as tfds


# splitting is wasting time, just append and prepend tokens
def preprocess(sent,eval=False):
    """returns a cleaned sentence with eos and sos tokens"""
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.replace("quot", "")
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    x = w.split(' ')
    if not eval:
        w = '<sos> ' + " ".join(x) + ' <eos>'
    return " ".join(x)



# make it a tf function

def limit_seq_length(seq,limit=0):

    seq=seq.split()

    if limit==0:
        return '<sos> ' + " ".join(seq) + ' <eos>'
    else:
        return '<sos> ' + " ".join(seq[:limit]) + ' <eos>'



def clean(s):
    # vymazat 039 je hoax
    w = re.sub(r'[\s+]'," ",s)
    w = re.sub(r'[" "]+', " ", w)
    w = w.replace("&quot;", "")
    w = w.replace("\n", "")
    w = w.translate(str.maketrans('', '', '\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.strip()
    return w


def get_data_lists(args,test=False,n=100):
    if test:
        with open("new_input","r") as i, open("new_target","r") as t:
            input,target = i.readlines(),t.readlines()
            return input[args.max_sentences:args.max_sentences+n],target[args.max_sentences:args.max_sentences+n]

    with open("new_input","r") as i, open("new_target","r") as t:
        input,target = i.readlines(),t.readlines()
        return input[:args.max_sentences],target[:args.max_sentences]



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
        with open("new_input","w") as input, open("new_target","w") as target:
            for clanok in data:
                headline=clean(clanok[0]["title"])
                for koment in clanok:
                    input.write(headline)
                    input.write('\n')
                    target.write(clean(koment["reaction"]))
                    target.write('\n')


if __name__ == "__main__":
    pass
