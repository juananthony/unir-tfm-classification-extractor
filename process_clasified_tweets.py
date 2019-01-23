from pymongo import MongoClient
import os
import csv

def saveClassification(filename, tweets):
    with open(filename + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["classified", "text", "created_at", "timestamp_ms"])
        for tweet in tweets:
            writer.writerow([tweet['classified'], getTweetText(tweet), tweet['created_at'], tweet['timestamp_ms']])

def getTweetText(tweet):
    if hasattr(tweet, 'extended_tweet'):
        return tweet['extended_tweet']['full_text']
    else:
        return tweet['text']

COLLECTION_NAME = 'classify'
PROJECTION_FILTER = {"classified": True, "text": True, "timestamp_ms": True, "created_at": True, 'extended_tweet.full_text':True}

MONGO_HOST = 'mongodb+srv://' + os.environ['MONGO_USER'] + ':' + os.environ['MONGO_PASS'] + '@' + os.environ['MONGO_SERVER'] + '/test?retryWrites=true'

# Creating MongoDB client connected to configured host by system properties
databaseClient = MongoClient(MONGO_HOST)
# Use metrotwitterdb database. If it doesn't exist, it will be created.
database = databaseClient.classification

issues = database[COLLECTION_NAME].find({'classified': 'issue'}, PROJECTION_FILTER)
complaints = database[COLLECTION_NAME].find({'classified': 'complaint'}, PROJECTION_FILTER)
nothings = database[COLLECTION_NAME].find({'classified': 'nothing'}, PROJECTION_FILTER)

saveClassification("issues", issues)
saveClassification("complaints", complaints)
saveClassification("nothings", nothings)
