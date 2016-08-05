
import itertools
import nltk

from utils import *
from RNNClass import RNNTheano
from nltk.corpus import wordnet #for calculating similarity
import distance #also for calculating similarity

_VOCABULARY_SIZE = 6500
_HIDDEN_DIM = 80
_LEARNING_RATE = 0.005
_NEPOCH = 100

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print ("Reading CSV file...")
with open('data/harryPotter_NoEmptyLines.txt', 'rb') as f:

    reader = f.readlines()


    # Split into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x.decode('utf-8').lower()) for x in reader])

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








model = RNNTheano(vocabulary_size, hidden_dim=80)

load_model_parameters_theano('./data/HarryPotterModel_Final.npz', model)




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





#computes jaccard and levinstein distance using python pacages "Distance"
def calculateSimilarity_WithDistancePackage(createdSentence): #createdSentence is list of words

    levinDist = {}
    jaccardDist = {}
    bestValues = {}
    for i in range(len(X_train)):
        currSentence = X_train[i]
        sentence_str = [index_to_word[x] for x in currSentence[1:-1]]  # sentence_str is list of words
        #Levinstein Distance
        dist = distance.levenshtein(createdSentence, sentence_str)
        dist2 = distance.jaccard(createdSentence,sentence_str)
        #print(dist)
        if (dist>0):
            #print ("Distance Levinshtein: %f" % (dist))
            levinDist[i]=dist
        jaccardDist[i]=dist2
        #print ("Jaccard Distance: %f" % (dist2))

    #take best value
    levinMin = min(levinDist.itervalues())
    jaccardMin = min(jaccardDist.itervalues())

    print ("Best Distance Levinshtein: %f" % (levinMin))
    print ("Best Distance Jaccard: %f" % (jaccardMin))
    bestValues["Jaccard"]=jaccardMin
    bestValues["Levin"]=levinMin
    return bestValues







#generate sentences and calculating the jaccard and levinstein similarity
def createSentences(numOfSentences):
    sumLevin=0
    sumJaccard=0
    for i in range(numOfSentences):
        sent = []
        sent = generate_sentence(model) # sent is list of words
        print " ".join(sent)
        distances = calculateSimilarity_WithDistancePackage(sent)
        sumLevin=sumLevin+distances["Levin"]
        sumJaccard=sumJaccard+distances["Jaccard"]

    print ("Avarage of minimum distances of jaccard: %f" % (sumJaccard/numOfSentences))
    print ("Avarage of minimum distances of levinsterin: %f" % (sumLevin / numOfSentences))





#calculates the avarage loss over all sentences in the data
def calculateSimilarity():
    totalLoss =0
    for i in range(len(y_train)): #iterate over each sentence and calculate the loss
        currSentence_Loss = model.calculate_loss_sentence(X_train[i],y_train[i])
        totalLoss = totalLoss+currSentence_Loss
        #print("Sentence number %d and loss for the sentence is %f." % (i, currSentence_Loss))
    print ("Average loss for all sentences in the text: %f" %(totalLoss/len(y_train)))

def getSentenceFromIndices(sentence):
    sentence_str = [index_to_word[x] for x in sentence[1:-1]] #sentence_str is list of words
    ans=""
    for i in range(len(sentence_str)):
        ans = ans+ " " + sentence_str[i]
    return ans


def calculateSimilarity_Random(amount):
    from random import randint
    totalLoss=0
    for i in range(amount): #pick #amount random sentences
        randomNum = randint(0,len(y_train)) #pick a random sentence from the text
        currSentence_Loss = model.calculate_loss_sentence(X_train[randomNum],y_train[randomNum])
        sentence = getSentenceFromIndices(X_train[randomNum])
        print("Sentence Number %d:[ %s ] Loss: %f" % (randomNum, sentence, currSentence_Loss))
        totalLoss=totalLoss+currSentence_Loss

    print("Average loss for the random sentences is: %f" % (totalLoss / amount))

#prediction = model.predict(X_train[3])
#sentence_str = [index_to_word[x] for x in prediction[1:-1]]



calculateSimilarity()
calculateSimilarity_Random(50) #pick 50 random sentences and calculate the loss


num_sentences = 50
createSentences(num_sentences)



