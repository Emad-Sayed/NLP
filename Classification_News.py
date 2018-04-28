import random
import nltk
import collections

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
word_features = list(all_words.keys())[:1000]



def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set= featuresets[1200:]
test_set = featuresets[:200]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set)*100)
classifier.show_most_informative_features(8)

