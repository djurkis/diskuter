#!/usr/bin/env python3

import sys
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import itertools

random.seed(420)

def get_ids_of_best_matches(embeddings,queries):
    """
    returns list of indexes of training headlines most similiar to queries
    """
    res = cosine_similarity(embeddings,queries)
    best_matches = []
    for col in range(res.shape[1]):
        best_matches.append(np.argmax(res[:, col]))

    return best_matches



def get_train_test_headlines(indexes,embedding_type):
    with open("train_uniqs", 'r') as f:
        train_x = f.readlines()
        train_x = [x.strip() for x in train_x]

    most_similiar = [train_x[x] for x in indexes]

    with open("train.src", 'r') as f:
        train_x = f.readlines()
        train_x = [x.strip() for x in train_x]


    with open("train.tgt", 'r') as f:
        train_y = f.readlines()
        train_y = [x.strip() for x in train_y]

    lookup={}
    for i,line in enumerate(train_x):
        if line in lookup:
            lookup[line].append(train_y[i])
            continue
        lookup[line]=[train_y[i]]

    with open("test.src", 'r') as f:
        test_x = f.readlines()
        test_x = [x.strip() for x in test_x]
        # print(len(test_x))
    # groups = itertools.groupby(test_x)

    grouped_src=[list(y) for x,y in itertools.groupby(test_x)]
    group_lens = [len(x) for x in grouped_src]

    # print(sum(group_lens))
    # sys.exit(0)
    with open("best.trans.sents", 'r') as f:
        b = f.readlines()
        b = [x.strip() for x in b]

    with open("marian_out","w") as f:
    # with open("f_cosine_predictions_"+embedding_type+"_random_7k", 'w') as f:

        for q,group,generated in zip(most_similiar,grouped_src,b):
            # print(most_similiar)
            cardinality=len(group)
            comment = generated
            # random

            for _ in range(cardinality):
                # comment = random.choice(lookup[q])
                f.write(comment)
                f.write('\n')
            # norandom
            # sofar=0
            # for comment in itertools.cycle(lookup[q]):
            #
            #     f.write(comment)
            #     f.write('\n')
            #     sofar+=1
            #     if sofar==cardinality:
            #         break
    # sys.exit(0)



if __name__ == '__main__':



# queries
    test_means = np.load("test_means.npz")['arr_0']
    test_maxes = np.load("test_maxes.npz")['arr_0']
    test_means2 = np.load("test_means2.npz")['arr_0']
    test_maxes2 = np.load("test_maxes2.npz")['arr_0']
    test_means3 = np.load("test_means3.npz")['arr_0']
    test_maxes3 = np.load("test_maxes3.npz")['arr_0']
    # print(test_means.shape)
    # sys.exit(0)
    queries=[test_means, test_maxes, test_means2, test_maxes2, test_means3, test_maxes3]
# db
    means = np.load("train_means.npz")['arr_0']
    maxes = np.load("train_maxes.npz")['arr_0']
    means2 = np.load("train_means2.npz")['arr_0']
    maxes2 = np.load("train_maxes2.npz")['arr_0']
    means3 = np.load("train_means3.npz")['arr_0']
    maxes3 = np.load("train_maxes3.npz")['arr_0']
    data = [means, maxes, means2, maxes2, means3, maxes3]
    names = ['means', 'maxes', 'means2', 'maxes2', 'means2', 'maxes3']

    for name, (e,q) in zip(names, zip(data,queries)):
        res = get_ids_of_best_matches(e,q)
        get_train_test_headlines(res,name)
