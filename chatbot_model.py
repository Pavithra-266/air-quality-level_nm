 import json
import random
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

vectorizer = CountVectorizer()
corpus, tags = [], []

for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

X = vectorizer.fit_transform(corpus)
clf = MultinomialNB()
clf.fit(X, tags)

def get_response(user_input):
    inp_vect = vectorizer.transform([user_input])
    tag = clf.predict(inp_vect)[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."
