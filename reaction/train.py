#!/usr/bin/env python3
import numpy
import pickle
import re

import sys

# ["title" : "fico na hrad" , comments : [,,,,,] ]


# from half_filtered


with open("half_filtered",'rb') as h:
    data = pickle.load(h)


dics = [{"title": x[0]['title'], "comments": x[0]['reaction']}
        for x in data]


vocab = [ x['title'] + x['comments'] for x in dics ]
vocab = sorted(set([x for y in vocab for x in y]))


# zmenit chars na 100 most used
from collections import Counter

def preprocess(corpus):
    count={}
    for atom in corpus:
        chars=atom["title"]+atom["comments"]
        for char in chars.lower():
            if char in count:
                count[char]+=1
            else:
                count[char]=1
    return Counter(count)

a =preprocess(dics)

charvocab = [x for x,y in a.most_common(61)]

def _2integer(s,lang):
    res=[]
    for char in s:
        if char in lang.char2index:
            res.append(lang.char2index[char])
        else:
            res.append(0)
    return res

def clean(corpus,lang):
    res = []
    for atom in corpus:
        title= _2integer(atom["title"],lang)
        comment= _2integer(atom["comments"],lang)
        res.append({"title":title,"comments":comment})
    return res

class Language:
    def __init__(self,voc):
        self.char2index={"<UNK>":0,"<SOS>":1,"<EOS>":2}
        for i,c in enumerate(voc):
            self.char2index[c]=i+3
        self.char2count={}
        self.index2char={0:"<UNK>",1:"<SOS>",2:"<EOS>"}
        for i in range(len(voc)):
            self.index2char[i+3]=voc[i]


# lang is a collection of  charmaps
lang = Language(charvocab)


integers = clean(dics,lang)

# sem niekde pridat SOS a EOS

# berie len prve vety for simplicity



# TODO: clean punctuation, remove sparse chars?, START and END token

# TODO: tf.keras.preprocessing.pad_sequences(   )

seq_length = 100
examples_per_epoch = 400 // seq_length
