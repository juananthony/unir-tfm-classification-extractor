from pymongo import MongoClient
from itertools import chain
import os
import csv
import nltk
import pickle
import time
import math
from random import shuffle
from nltk.tokenize import word_tokenize

start = time. time()

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

def saveClassification(filename, tweets):
    with open(filename + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["classified", "text"])
        for tweet in tweets:
            writer.writerow([tweet['classified'], getTweetText(tweet)])

def getTweetText(tweet):
    if 'extended_tweet' in tweet:
        return tweet['extended_tweet']['full_text']
    else:
        return tweet['text']

def getEntry(tweet):
    return (getTweetText(tweet), tweet['classified'])

def saveClassifier(classifier, all_words):
    f = open('naive_bayes.classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    f = open('naive_bayes.all_words.pickle', 'wb')
    pickle.dump(all_words, f)
    f.close()

PERCENTAGE_TRAIN = 0.8

COLLECTION_NAME = 'classify'
PROJECTION_FILTER = {"classified": True, "text": True, 'extended_tweet.full_text':True}

MONGO_HOST = 'mongodb+srv://' + os.environ['MONGO_USER'] + ':' + os.environ['MONGO_PASS'] + '@' + os.environ['MONGO_SERVER'] + '/test?retryWrites=true'

print("Getting tweets from Database . . .")

# Creating MongoDB client connected to configured host by system properties
databaseClient = MongoClient(MONGO_HOST)
# Use metrotwitterdb database. If it doesn't exist, it will be created.
database = databaseClient.classification

issues_list = list(database[COLLECTION_NAME].find({'classified': 'issue'}, PROJECTION_FILTER))
complaints_list = list(database[COLLECTION_NAME].find({'classified': 'complaint'}, PROJECTION_FILTER))
nothings_list = list(database[COLLECTION_NAME].find({'classified': 'nothing'}, PROJECTION_FILTER))

print("Saving tweets into csv files . . .")

saveClassification("issues", issues_list)
saveClassification("complaints", complaints_list)
saveClassification("nothings", nothings_list)

print("Extracting text from tweets . . .")

issues = []
for tweet in issues_list:
    issues.append(getEntry(tweet))
complaints = []
for tweet in complaints_list:
    complaints.append(getEntry(tweet))
nothings = []
for tweet in nothings_list:
    nothings.append(getEntry(tweet))
shuffle(issues)
shuffle(complaints)
shuffle(nothings)

print("Joining words from all tweets . . .")
min_examples = min([len(issues), len(complaints), len(nothings)])
print("min examples: " + str(min_examples))
all_tweets = issues[:min_examples] + complaints[:min_examples] + nothings[:min_examples]
shuffle(all_tweets)

print("Tweet number: " + str(len(all_tweets)))
limit_train = math.ceil(len(all_tweets) * PERCENTAGE_TRAIN)
print("tweets to train:" + str(limit_train))
all_tweets_train = all_tweets[:limit_train]
all_tweets_test = all_tweets[(limit_train + 1):]

#print(all_tweets_train)
print("Getting all words . . .")
all_words = set(word.lower() for tweet in all_tweets_train for word in word_tokenize(tweet[0])) - set(nltk.corpus.stopwords.words("spanish"))

print("Creating tuples . . .")
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in all_tweets_train]

print("Training classifier . . .")
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()
saveClassifier(classifier, all_words)

print("Testing classifier . . .")

num_correct_issues = 0
num_correct_complaints = 0
num_correct_nothings = 0
num_incorrect_issues = 0
num_incorrect_complaints = 0
num_incorrect_nothigs = 0
num_tests = 0

for tweet in all_tweets_test:
    test_tweet_features = {word.lower(): (word in word_tokenize(tweet[0].lower())) for word in all_words}

    #test_tweet_features = set(test_tweet_features) - set(nltk.corpus.stopwords.words("spanish"))

    classification = classifier.classify(test_tweet_features)

    print("------------------------------------------------------------")
    print("Tweet: " + tweet[0])
    print("Class: " + tweet[1])
    print("Classification: " + classification)

    num_tests += 1

    if(tweet[1] == classification):
        if(classification == "issue"):
            num_correct_issues += 1
        if(classification == "complaint"):
            num_correct_complaints += 1
        if(classification == "nothing"):
            num_correct_nothings += 1
    else:
        if(classification == "issue"):
            num_incorrect_issues += 1
        if(classification == "complaint"):
            num_incorrect_complaints += 1
        if(classification == "nothing"):
            num_incorrect_nothigs += 1
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print(" - Number of tweets for testing: " + str(num_tests))
print(" - Number of correct tweets classified as ISSUE: " + str(num_correct_issues))
print(" - Number of inccorrect tweets classified as ISSUE: " + str(num_incorrect_issues))
print(" - Number of correct tweets classified as COMPLAINT: " + str(num_correct_complaints))
print(" - Number of incorrect tweets classified as COMPLAINT: " + str(num_incorrect_complaints))
print(" - Number of correct tweets classified as NOTHING: " + str(num_correct_nothings))
print(" - Number of Iinorrect tweets classified as NOTHING: " + str(num_incorrect_nothigs))
print("------------------------")
print("% accuracy: " + str(((num_correct_issues + num_correct_complaints + num_correct_nothings) / num_tests) * 100) + " %")

end = time.time()
print(" >>> %s minutes" % (end - start)/60)
