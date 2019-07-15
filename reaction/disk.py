#!/usr/bin/env python3
import numpy as np
import pickle
import re
import tensorflow as tf
import sys



with open("half_filtered",'rb') as h:
    data = pickle.load(h)



def preprocess(sent):
    w = sent.lower()
    w = re.sub(r"([?,.!])", r" \1 ",w)
    w = re.sub(r'[" "]+'," ",w)
    w = w.translate(str.maketrans('','','\"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'))
    w = w.rstrip().strip()
    w = '<SOS> ' + w + ' <EOS>'
    return w

data = data[:5000]
titulky = []
komentare = []
for clanok in data:
    for koment in clanok:
        titulky.append(preprocess(koment["title"]))
        komentare.append(preprocess(koment["reaction"]))


token = tf.keras.preprocessing.text.Tokenizer(filters='')
token.fit_on_texts(titulky)
token.fit_on_texts(komentare)

tensor_titulky = token.texts_to_sequences(titulky)
tensor_koment = token.texts_to_sequences(komentare)

seqs_titulky = tf.keras.preprocessing.sequence.pad_sequences(tensor_titlky,padding="post")
seqs_koment = tf.keras.preprocessing.sequence.pad_sequences(tensor_koment,padding="post")
