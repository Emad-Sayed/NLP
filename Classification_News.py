import random
import nltk
import collections
from nltk.classify import DecisionTreeClassifier

list_ = []
words = []
documents = []
SingleDocument = [[], ""]
for num in range(1, 510):
    list_ = []
    with open("business" + "/" + str(num) + ".txt", 'r') as f:
        for line in f:
            for word in line.split():
                list_.append(word)
                words.append(word)
        SingleDocument = (list_, "business")
        documents.append(SingleDocument)

for num in range(1, 510):
    list_ = []
    with open("sport" + "/" + str(num) + ".txt", 'r') as f:
        for line in f:
            for word in line.split():
                list_.append(word)
                words.append(word)
        SingleDocument = (list_, "sport")
        documents.append(SingleDocument)

for num in range(1, 386):
    list_ = []
    with open("entertainment" + "/" + str(num) + ".txt", 'r') as f:
        for line in f:
            for word in line.split():
                list_.append(word)
                words.append(word)
        SingleDocument = (list_, "entertainment")
        documents.append(SingleDocument)

random.shuffle(documents)
print(len(documents))
print(len(words))

all_words = nltk.FreqDist(words)
word_features = list(all_words.keys())[:800]



def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set= featuresets[:1000]
test_set = featuresets[1000:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("NaiveBayesClassifier Accuracy     =>"+str(nltk.classify.accuracy(classifier, test_set)*100))
# To Test This Application Put in File 1.txt and try to make the text large as possible because the features not large (small data set)
#InputList=[]
#with open("1.txt", 'r') as f:
#    for line in f:
#        for word in line.split():
#            InputList.append(word)
#            words.append(word)
#print(classifier.classify(document_features(InputList)))

