# Diskuter

Diskuter was created with the goal to examine deep learning techniques in text generation.


### Contents

* [Architecture](#Architecture)
* [Data](#Data)
* [Direction](#Direction)
* [Installation](#Installation)
* [Results](#Results)
* [Sources](#Sources)



## Architecture

The core of the baseline model architecture is an encoder-decoder RNN with attention.

[Attention](https://arxiv.org/abs/1508.04025) seems to help solve the bottleneck of information flow from longer sequences.

<!-- ![attention](/readme_data/attention_mechanism.jpg?raw=true "attention") -->

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




## Sources



## Results

TODO...


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




## Direction
After experimenting and getting a feel for working with Tensorflow the next steps will be aimed at using
[tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.
