import json
import pandas as pd
import pickle
import re
import nltk
import numpy
import string
import keras
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential

from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model


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

#get colloquial-indonesian-lexicon.csv to a variable
normalized_dict = pd.read_csv('normalisasi - colloquial-indonesian-lexicon.csv').set_index('slang')['formal'].to_dict()

def normalized(text, norm_dict):
    tokenize = [word for word in text.split()]
    return " ".join([norm_dict[term] if term in norm_dict else term for term in tokenize])


with open("dataset-train.json", encoding="utf8") as file:
    data = json.load(file)

with open("Dataset-test.json", encoding="utf8") as file:
    data_test = json.load(file)

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




#make training pickle
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

#make testing pickle
try:
    with open("chatbot_test.pickle", "rb") as file:
        words, labels, X_test, Y_test = pickle.load(file)
    print("data_test is ready")

except:

    X_test = []
    Y_test = []
    Y_test_empty = [0 for _ in range(len(labels))]

    for intent in data_test["intents"]:
        for pattern in intent["patterns"]:
            bow_data_test = bag_of_words(pattern, words)
            X_test.append(bow_data_test)
            label_test = intent["tag"]

        for a, b in list(enumerate(labels)):
            if label_test == b:
                Y_test_row_test = Y_test_empty[:]
                Y_test_row_test[a] = 1
                Y_test.append(Y_test_row_test)

    X_test = numpy.array(X_test)
    Y_test = numpy.array(Y_test)

    with open("chatbot_test.pickle", "wb") as file:
        pickle.dump((words, labels, X_test, Y_test), file)

#training proccess
try:
    with open('chatbotmodel.json') as json_file:
        print(type(json_file))
        model_json = json.load(json_file)
    myChatModel = keras.models.model_from_json(model_json)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Loaded model from disk")

except:
    # Make our BPNN model
    myChatModel = Sequential()
    myChatModel.add(Dense(128, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dropout(0.50))
    myChatModel.add(Dense(64, activation='relu'))
    myChatModel.add(Dropout(0.50))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    # optimize the model
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    myChatModel.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])

    # train the model
    history = myChatModel.fit(training, output, epochs=600, batch_size=100, verbose=1, validation_data=(X_test, Y_test))

    # serialize model to yaml and save it to disk
    model_json = myChatModel.to_json()
    with open("chatbotmodel.json", "w") as y_file:
        json.dump(model_json, y_file)

    # serialize weights to HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Saved model from disk")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


score = myChatModel.evaluate(training, output, verbose=1)

print("Training Loss:", score[0])
print("Training Accuracy:", score[1])


score = myChatModel.evaluate(X_test, Y_test, verbose=1)

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
