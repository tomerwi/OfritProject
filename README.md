# FinalProject

1.  

### Collection of data sequences

The collection of data sequences that we choose is Harry Potter And The Chamber Of Secrets that wrriten by by J. K. Rowling. This is book 2 in the Harry Potter series.

2.

### Description of data

We chose this topic because we both really love this topic. We both read the book series of Harry Potter as well as the movie came out we saw. Books and films very fascinated us and so we found the contents of the book text file we knew that this was the theme that we want to do about the project.

The main challenge in dealing with this topic is the investigation of the words are not real words. Books of Harry Potter there are many phrases and words they use them as "magic" that these words are a lot for the book.
In addition, in the past we never study this kind of analysis sequences so that also make it tougher on us.

3.

### Preprocessing

vocabulary_size = 7000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

#### comment: We read the data and append SENTENCE_START and SENTENCE_END tokens

with open('data/harrypotter.txt', 'rb') as f:
reader = f.readlines()

#### comment: Split into sentences

sentences = itertools.chain(*[nltk.sent_tokenize(x.decode('utf-8').lower()) for x in reader])

#### comment: Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print ("Parsed %d sentences." % (len(sentences)))

#### comment: Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

#### comment: Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))

#### comment: Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

#### comment: Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

#### comment: Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#### comment: Init the model with random values for U, S, V. This function is attached to the project code.

model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)

#### comment: Start Training the model
train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)



