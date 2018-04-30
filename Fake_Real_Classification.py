import random
import nltk
import collections
list_ = []
words = []
documents = []
SingleDocument = [[], ""]
for num in range(1, 2000):
    list_ = []
    with open("Real_Fake_News/fake" + "/" + str(num) + ".txt", 'r') as f:
        for line in f:
            for word in line.split():
                list_.append(word)
                words.append(word)
        SingleDocument = (list_, "fake")
        documents.append(SingleDocument)
for num in range(1, 2000):
    list_ = []
    with open("Real_Fake_News/real" + "/" + str(num) + ".txt", 'r') as f:
        for line in f:
            for word in line.split():
                list_.append(word)
                words.append(word)
        SingleDocument = (list_, "real")
        documents.append(SingleDocument)
print(len(documents))
random.shuffle(documents)
print(len(documents))
print(len(words))
all_words = nltk.FreqDist(words)
print(all_words)
word_features = list(all_words.keys())[:2500]
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set= featuresets[:3000]
test_set = featuresets[3000:]