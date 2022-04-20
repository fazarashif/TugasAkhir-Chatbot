import json
import pandas as pd
import numpy as np
import pickle
import random
import re
import nltk
import numpy
import string
import keras
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_yaml

#from keras.optimizers import Adam

from mysql.connector import connect


import random

nltk.download('punkt')
nltk.download('stopwords')


stemmer_sastrawi = StemmerFactory().create_stemmer()

def text_preprocess(text):
    text = str(text) #convert to string
    text = text.lower() #Convert to lower case
    text = text.encode('ascii', 'replace').decode('ascii') #Remove non ascii
    text = text.strip() #Remove whitespace
    text = text.translate(str.maketrans('','',string.punctuation)) #Remove punctuation
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','',text) #Remove links
    text = " ".join(re.split(r"\s+", text)) #Remove additional white spaces
    text = re.sub('[\s]+', ' ', text) #Remove additional white spaces
    text = re.sub(r"\d+", "", text) #Remove number
    text = re.sub(r'\b[a-zA-Z]\b', '', text) #Remove single char
    text = text.strip('\'"') #trim
    return text

#Get stopwords for Bahasa Indonesia
list_stopwords = stopwords.words('indonesian')
list_needwords = ["tidak", "ada"]
for i in list_needwords:
    if i in list_stopwords:
        list_stopwords.remove(i)

def remove_stopwords(text):
    message = [word for word in text.split() if len(word)>1 and word not in list_stopwords]
    return " ".join(message)


normalized_dict = pd.read_csv('normalisasi - colloquial-indonesian-lexicon2.csv').set_index('slang')['formal'].to_dict()

def normalized(text, norm_dict):
    tokenize = [word for word in text.split()]
    return " ".join([norm_dict[term] if term in norm_dict else term for term in tokenize])


with open("dataset-baru6.json", encoding="utf8") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            pattern = text_preprocess(pattern)
            pattern = normalized(pattern, normalized_dict)
            pattern = remove_stopwords(pattern)

            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer_sastrawi.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer_sastrawi.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)


try:
    with open('chatbotmodel.json') as json_file:
        print(type(json_file))
        model_json = json.load(json_file)
    myChatModel = keras.models.model_from_json(model_json)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Loaded model from disk")

except:
    # Make our neural network
    myChatModel = Sequential()
    myChatModel.add(Dense(128, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dropout(0.5))
    myChatModel.add(Dense(64, activation='relu'))
    myChatModel.add(Dropout(0.5))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    # optimize the model
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    myChatModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    myChatModel.fit(training, output, epochs=1000, batch_size=8, verbose=1)

    # serialize model to yaml and save it to disk
    model_json = myChatModel.to_json()
    with open("chatbotmodel.json", "w") as y_file:
        json.dump(model_json, y_file)

    # serialize weights to HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model from disk")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s = text_preprocess(s)
    s = remove_stopwords(s)
    s = normalized(s, normalized_dict)

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer_sastrawi.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


# Adding some context to the conversation i.e. Contexualization for altering question and intents etc.
# create a data structure to hold user context
# context = {}
#
# ERROR_THRESHOLD = 0.30
#
#
# def chatWithBot(inputText, userID='123'):
#     # generate probabilities from the model
#     currentText = bag_of_words(inputText, words)
#     currentTextArray = [currentText]
#     numpyCurrentText = numpy.array(currentTextArray)
#
#     if numpy.all((numpyCurrentText == 0)):
#         return "maaf, chatbot masih belum mengerti. Tolong sampaikan gejala penyakit yang dialami anak."
#
#     results = myChatModel.predict(numpyCurrentText[0:1])
#     # return 2dimensional array to 1 dimensional array
#     results = results.flatten()
#
#     # filter out predictions below a threshold
#     results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append((labels[r[0]], r[1]))
#     # return tuple of intent and probability
#     results = return_list
#
#     # if we have a classification then find the matching intent tag
#     if results:
#         # loop as long as there are matches to process
#         while results:
#             for i in data['intents']:
#                 # find a tag matching the first result
#                 if i['tag'] == results[0][0]:
#                     # set context for this intent if necessary
#                     if (userID in context and 'context_filter' in i and 'context_set' in i and i['context_filter'] ==
#                             context[userID]):
#                         context[userID] = i['context_set']
#
#                         responses = i['responses']
#                         return random.choice(responses)
#
#                     if 'context_set' in i:
#                         context[userID] = i['context_set']
#
#                     # check if this intent is contextual and applies to this user's conversation
#                     if not 'context_filter' in i or \
#                             (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
#                         # a random response from the intent
#
#                         responses = i['responses']
#                         return random.choice(responses)
#
#             results.pop(0)

#ini coba chatwith bot
# Adding some context to the conversation i.e. Contexualization for altering question and intents etc.
# create a data structure to hold user context
context = {}
fallback_intent = "Mohon maaf ya, Chatbot belum bisa menjawab pertanyaan anda dan masih perlu banyak belajar nih! Mohon ceritakan pertanyaan atau gejala yang dialami anak ya!"
ERROR_THRESHOLD = 0.10


def chatWithBot5(inputText, userID='123'):
    # generate probabilities from the model
    currentText = bag_of_words(inputText, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    results = myChatModel.predict(numpyCurrentText[0:1])
    # return 2dimensional array to 1 dimensional array
    results = results.flatten()

    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((labels[r[0]], r[1]))

    results = return_list
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in data['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if (userID in context and 'context_filter' in i and 'context_set' in i and i['context_filter'] ==
                            context[userID]):
                        context[userID] = i['context_set']

                        responses = i['responses']
                        return random.choice(responses)

                    if ('context_set' in i) and (not 'context_filter' in i):
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        # a random response from the intent

                        responses = i['responses']
                        return random.choice(responses)

            results.pop(0)
        return fallback_intent

    else:
        return fallback_intent

def chatbotResponse(inputText, chatUserID='123'):
    db = connect(
        host="127.0.0.1",
        port="3306",
        user="root",
        passwd="",
        database="intentchatbot")

    cursor_db = db.cursor()

    userID = chatUserID
    response = chatWithBot5(inputText, chatUserID)

    if response == fallback_intent:
        fallback = '1'
    else:
        fallback = '0'

    val = (userID, inputText, fallback)
    query = "INSERT INTO intents (userID, intent, fallback_flag) VALUES (%s, %s, %s)"

    cursor_db.execute(query, val)
    db.commit()

    if inputText.lower() == "keluar":
        response = "apakah anda cukup terbantu setelah bercakap dengan chatbot ini ?"
        context[userID] = "feedback"

    cursor_db.close()

    return response

# def chat(chatUserID):
#     print(
#         "Selamat datang di chatbot, silahkan sampaikan gejala yang dialami oleh anak.. (ketik keluar jika ingin berhenti berbicara dengan chatbot)")
#
#     db = connect(
#         host="127.0.0.1",
#         port="3306",
#         user="root",
#         passwd="",
#         database="intentchatbot")
#
#     cursor_db = db.cursor()
#
#     while True:
#         inp = input("You: ")
#         userID = chatUserID
#         response = chatWithBot(inp, chatUserID)
#
#         if response == "maaf, chatbot masih belum mengerti. Tolong sampaikan gejala penyakit yang dialami anak.":
#             fallback = '1'
#         else:
#             fallback = '0'
#
#         val = (userID, inp, fallback)
#         query = "INSERT INTO intents (userID, intent, fallback_flag) VALUES (%s, %s, %s)"
#
#         cursor_db.execute(query, val)
#         db.commit()
#
#         if inp.lower() == "keluar":
#             print("apakah anda cukup terbantu setelah bercakap dengan chatbot ini ?")
#             context[userID] = "feedback"
#             inp2 = input("You: ")
#             response2 = chatWithBot(inp2, chatUserID)
#             print(response2)
#
#             break
#
#         print(response)
#
#     cursor_db.close()

#   chat("test1")