# Diskuter

Diskuter was created with the goal to examine deep learning techniques in text generation.

The problem outline:
Given Headline about an article , output a discussion commentary.

The Metric for measuring the commentaries is not well defined, but its semantics should be correlated with the semantics of the article.
Also some form of naturality, is desired. Commentary that looks like it was written by a human would be preferable.

One way, how to achieve this is to reuse ideas from machine translation and model a distribution P and pick  output that maximizes P(output|input).


### Contents

* [Architecture](#Architecture)
* [Data](#Data)
* [Direction](#Direction)
* [Installation](#Installation)
* [Results](#Results)
* [Sources](#Sources)



## High level architecture

We start by having an input and a target file. Each is a newline delimited string corresponding to headline and a discussion commentary from SME.sk.

Firstly, a tokenizer tokenizes strings into words, and then translates them into integers creating a dictionary.
Line pairs are then batched and padded so they are uniform in length.



Diskuter consists of an Encoder and a Decoder.
Encoder recieves input sequence of integers and embeds it into real valued vector, does its computation and outputs a context vector, which is a representation of the input sequence.

![attention](/readme_data/attention_mechanism.jpg?raw=true "attention")

Decoder takes this representation and autoregressively outputs a softmax over the vocabulary at each timestep, and finally returns a sequence of words that are 'most' probable. Teacher forcing is used to speed up the training. (Using golden labels for the decoder input instead of possibly erroneous ones)




[Attention](https://arxiv.org/abs/1508.04025) seems to help solve the bottleneck of information flow from longer sequences.

#### Embedding

Due to the nature of natural language and its categorical nature, we need to project sequences of strings into sequences of real valued vectors.

Diskuter has an embedding layer in its Encoder and trains it during training.

It is possible to use pretrained embeddings such as [Fast-text](https://fasttext.cc), which are available as a word vector pairs.

Another option is to use an external pretrained network (Eg. BERT) and pass data through it before encoding.


#### Decoding

Decoding is done after the encoder is finished with its inputs.
We supply the decoder with the final state of the encoder as well as <sos> token.
This architecture is Autoregressive, meaning the output is then fed into the decoder until some stopping condition is met.
We are using Beam decoding, which for some specified k searches k-best paths at the same time.



Possible stopping criteria worth considering: [Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and
Stopping Criteria for Neural Machine Translation](https://arxiv.org/pdf/1808.09582.pdf)


## Installation

Tested on python 3.6.8 and tensorflow 2.0

`pip install tensorflow==2.0.0-rc1`



The entry point is the `main.py` which has minor assumptions about location of the data, and can be used for training and inference.



## Sources

[Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)

[MFF UK Deep learning course](https://ufal.mff.cuni.cz/courses/npfl114/1819-summe)

[Efficient Estimation of Word Representations in
Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

[Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)



## Results

Some experiment results are stored in ``experiment_results``.
The Model seemed to be learning something, as is apparent from the loss over time.
But Due to small dictionary size (20k) and the nature of the input (many rare words) the <unks> were an easy choice for the network. Using subword units and increasing `dictionary_size` will most likely help the quality.

Also the Basic model was trained only on a 20k subset of the dataset and 50k steps.


![lot](/experiment_results/basic_/Figure_1.png?raw=true "attention")



## Data



Data used to train the model were obtained from [SME.sk](hhtps://sme.sk).

After some preprocessing the data is in following format:
```
  'id_parent_reaction': 26498917,

  'id_reaction': 26515628,
  'title': '68 tipov na výlety po Slovensku'}
  'reaction': 'človek chce ušetriť, tak musí ísť na dovolenku do zahraničia, je to smutné, ale je to tak. **** hotel v centre Londýna, je napr. lacnejší ako ubytovanie v Bešeňovej.',  
  'reaction_count': 14,
  'positive':3,
  'negative'4,
  'stamp_created': 1530121373,
  'subject': 'keĎ'
```

For training only comments reacting directly to the article are used.
After filtering the comments reacting to other comments there are 1.2 Million
title reaction pairs that are used as input and gold label for the network.

As of the reaction strings, they were cleaned by a basic replacement of
characters that are considered noise. (Eg. links)






## Direction


Using a Transformer architecture together with better embeddings, language model and decoding procedure,
it will be possible to significantly improve the results.





[tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.
