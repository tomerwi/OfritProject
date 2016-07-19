#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '50'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')





vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print ("Reading CSV file...")
with open('data/comments.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    #reader = f.readlines()
    reader.next

    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print ("Parsed %d sentences." % (len(sentences)))



# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])







# added tomer
model = RNNTheano(vocabulary_size, hidden_dim=50)

load_model_parameters_theano('./data/trained-model-theano.npz', model)


def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    # sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


num_sentences = 10000
senten_min_length = 7



for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words

    sent = generate_sentence(model)
    print " ".join(sent)


# for i in range(num_sentences):
#     sent = []
#     # We want long sentences, not sentences with one or two words
#     while len(sent) & lt; senten_min_length:
#         sent = generate_sentence(model)
#     print " ".join(sent)

