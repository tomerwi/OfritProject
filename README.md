# FinalProject

1.  

### Collection of data sequences

The collection of data sequences that we chose is Harry Potter And The Chamber Of Secrets that was written by by J. K. Rowling. This is the second book in Harry Potter's series

2.

### Description of data

This was actually not our first data that we tried to learn. We started this project working on a collection of speeches of Binyamin Netanyahu. We gathered and combined a lot of speeches, but when we tried to learn and train the model, we had difficulties to create reasonable sentences from it. We think the reason for that was because the collection of the speeches were no big enough. 
We then chose Harry potter topic because that data is very big and has variety in its content (number of words, number of unique words etc). We think that the model training can be much more efficent in this context. 

The main challenge in dealing with this topic is the investigation of the words which are not "real" words. Harry Potter's books contain many phrases and words that describe magic, which can be challenging to learn.
In addition, in the past we never studied this kind of sequences analysys so this is also chanllenging for us. 

Data Analysis:
12867 sentences.
7605 unique words

3.

### Preprocessing

#### comment: Inilize variables. Replace all the words which are not in our vocabulary with "Unknown_Token". Append Setence_Start and Sentence_End to the sentence - it is used becuase we want to "teach" the model which words open sentences and which one end them.
Leraning rate - regulzation parameter - mainly used to prevent overfitting.
Nepoch - number of iterations in the training phaze. 
Hidden_Dim - the memory of the network - making it bigger allows us to learn complex patterns in the data



    vocabulary_size = 6500
    
    unknown_token = "UNKNOWN_TOKEN"
    
    sentence_start_token = "SENTENCE_START"
    
    sentence_end_token = "SENTENCE_END"
    
    _HIDDEN_DIM = 80
    _LEARNING_RATE = 0.005
    _NEPOCH = 100




#### comment: Read the data to the memory and tokenize into sentences. We used NLTK Python library for the tokenizing. When we first tried to train the model, we saw that it predicts the word "SENTENCE_END" at the beginning of the sentence. After investigation We found out that the reason for that was because the text had a lot of empty lines, which has a bad influence on the predictions. Therfore, we removed all the empty lines from the text. (We did it outside the code)

    with open('data/harrypotter.txt', 'rb') as f:

    reader = f.readlines()

    #### comment: Split into sentences

    sentences = itertools.chain(*[nltk.sent_tokenize(x.decode('utf-8').lower()) for x in reader])

    #### comment: Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]


#### comment: Tokenize words in each sentence (for example - Please come here! ->> {please} {come} {here} {!}. 

    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]



#### comment: Count the word freqencies

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))





#### comment: Get the most common words (vocabulary size) and build index_to_word and word_to_index vectors. The input for the RNN are vectors, not Strings, so we create mapping between words and indices. 

    vocab = word_freq.most_common(vocabulary_size - 1)
    
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])



#### comment: Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):

    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

#### comment: Create the training data. X is the sentences, Y is the sentences shifted right by one position. In this structure the model knows which word comes after each word. for example - Y[3] is the next word for X[3] in the sentence. 

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#### comment: Init the model with random values for U, S, V. This function is attached to the project code. We used Theano for building the model. Theano is Python library that let you to define, optimize and evaluate mathematical expression, especially ones with multi-dimensional arrays. Because RNN are easily expressed with multi-dimensioanl arrays, Theano is a great fit. 


    model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)

#### comment: Training the model. We send the training sentences (X_train and Y_train) for theano. Moreover, we set the Epoch number - which is the number of iteration over all sentences, and also set the learning rate for the SGD. 
After every 5 iteration, we calculate the loss and we check if it decreases. This part (calculating the loss) is not necessary, its just an indication for us. 
Training the model took us alot of time due to the high number of sentences in Harry Potter's book. 



    def train_with_sgd(model, X_train, y_train, learning_rate, nepoch, evaluate_loss_after=5):

        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch): #one epoch means one iteration over all training examples
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print ("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
                #Save the parameters after the iteration
                save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time),
                                             model)
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1



#### comment: Loadin the model. This function gets the path to the model file, and loads the model parameters to the memory. after the loading we can use this model to predict sentences.  

    def load_model_parameters_theano(path, model):
        npzfile = np.load(path)
        U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
        model.hidden_dim = U.shape[0]
        model.word_dim = U.shape[1]
        model.U.set_value(U)
        model.V.set_value(V)
        model.W.set_value(W)
        print ("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))
        
        
#### comment: Generating sentences from the model. we generate only sentences that is bigger then 7 words. Sentece is represented by the model as vector of numbers so we need to convert it to a string using the index_to_word dictionary. 

    def generate_sentence(model):
        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token] or len(new_sentence)<7:
            next_word_probs = model.forward_propagation(new_sentence)
            sampled_word = word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
    
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str
        
#### comment: Calculating the similarity between the real sentences and the predicted sentences. The measure that we used for the calculation is Cross-Entroty-Loss. for each word in a sentence, it measures how far our prediction was from the real word. For example, if we have a sentence - "We are eating dinner with some friends" and we want to predict the word which comes after "dinner". the model generate a vector (in the size of the vocabulary) with probabliltes for each word. let's say the the word "with" has a probabilty of 0.8 to be the next word. so the distance of the prediction is Yn*Log(On). Yn equals 1, On eqauls 0.8. We do this calculation for every word in the sentence and sum it up. finally, we divide it by the number of words in the sentence to get avarage loss for each word. 
We calculated the loss for the whole database, and also calculated the loss for 100 sentences that we picked randomally. 

    def calculateSimilarity():
        totalLoss =0
        for i in range(len(y_train)): #iterate over each sentence and calculate the loss
            currSentence_Loss = model.calculate_loss_sentence(X_train[i],y_train[i])
            totalLoss = totalLoss+currSentence_Loss
            #print("Sentence number %d and loss for the sentence is %f." % (i, currSentence_Loss))
        return totalLoss/len(y_train)
    
    def calculateSimilarity_Random(amount):
        from random import randint
        totalLoss=0
        for i in range(amount): #pick #amount random sentences
            randomNum = randint(0,len(y_train)) #pick a random sentence from the text
            currSentence_Loss = model.calculate_loss_sentence(X_train[randomNum],y_train[randomNum])
            totalLoss=totalLoss+currSentence_Loss
        return totalLoss / amount
        
    def calculate_total_loss_sentence(self, X, Y):
        return self.ce_error(X, Y)

    def calculate_loss_sentence(self,X,Y):
        num_words = len(Y)
        return self.calculate_total_loss_sentence(X,Y) / float(num_words)        
        

