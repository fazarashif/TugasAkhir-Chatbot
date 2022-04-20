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
from flask import Flask
from mysql.connector import connect
import random

nltk.download('punkt')
nltk.download('stopwords')

stemmer_sastrawi = StemmerFactory().create_stemmer()


#Load Dataset
with open("dataset-baru6.json", encoding="utf8") as file:
    data = json.load(file)

#Load chatbot pickle
with open("chatbot.pickle", "rb") as file:
    words, labels, training, output = pickle.load(file)

#Load Chatbot model
with open('chatbotmodel.json') as json_file:
    print(type(json_file))
    model_json = json.load(json_file)
myChatModel = keras.models.model_from_json(model_json)
myChatModel.load_weights("chatbotmodel.h5")
print("Loaded model from disk")

#declare function

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


normalized_dict = pd.read_csv('normalisasi - colloquial-indonesian-lexicon.csv').set_index('slang')['formal'].to_dict()

def normalized(text, norm_dict):
    tokenize = [word for word in text.split()]
    return " ".join([norm_dict[term] if term in norm_dict else term for term in tokenize])

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

#function for generate response from the prediction
context = {}
fallback_intent = "Mohon maaf ya, Chatbot belum bisa menjawab pertanyaan anda dan masih perlu banyak belajar nih! Mohon ceritakan pertanyaan atau gejala yang dialami anak ya!"
ERROR_THRESHOLD = 0.10


def chatWithBot(inputText, userID='123'):
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
    response = chatWithBot(inputText, chatUserID)

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


#Routing Flask to App inventor
app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

@app.route('/')
def index():
    return ("sistem chabot")


@app.route('/chatbot/<username>/<question>')
def chatbot_response(question, username):
   response = chatbotResponse(question, username)
   return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)


