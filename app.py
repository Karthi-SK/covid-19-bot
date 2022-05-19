from flask import Flask, render_template, request, url_for, redirect
# from pymongo import MongoClient
# from bson.objectid import ObjectId
# from flask_ngrok import run_with_ngrok
# import tensorflow
import random
import nltk
import json, requests, ast
import numpy, pickle, tflearn
from keras.models import Sequential
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()

# client = MongoClient('localhost', 27017)

# db = client.flask_db # db name
# todos = db.kudos # collection name

app = Flask(__name__)
# run_with_ngrok(app)

data = open('./static/new_intents.json').read()
data=ast.literal_eval(data)
# model = pickle.load(open('./static/seq.pkl', 'rb'))
# model = tensorflow.keras.models.load_model('./static/seq_model')

# words = ["'m", "'s", '19', 'a', 'air', 'and', 'anyon', 'ar', 'awesom', 'be', 'breath', 'bye', 'can', 'caus', 'chat', 'compuls', 'consult', 'contact', 'coron', 'coronavir', 'could', 'covid', 'covid-19', 'dat', 'day', 'detail', 'difficul', 'diseas', 'district', 'do', 'doct', 'doe', 'emerg', 'for', 'get', 'good', 'goodby', 'gudielin', 'guidelin', 'hav', 'hello', 'help', 'helplin', 'hey', 'hi', 'hol', 'hospit', 'hostp', 'how', 'i', 'if', 'in', 'infect', 'is', 'lat', 'link', 'look', 'lookup', 'mand', 'mask', 'me', 'meet', 'myself', 'next', 'nic', 'numb', 'of', 'off', 'paty', 'peopl', 'precaut', 'precauty', 'prev', 'prop', 'protect', 'provid', 'reg', 'remedy', 'report', 'sars-cov-2', 'search', 'see', 'should', 'show', 'solv', 'spread', 'stat', 'support', 'symptom', 'thank', 'that', 'the', 'ther', 'through', 'til', 'tim', 'to', 'transf', 'transmit', 'typ', 'up', 'us', 'vac', 'vaccin', 'vir', 'want', 'we', 'wear', 'what', 'when', 'wis', 'with', 'you']
# labels = ['Covid-19', 'Covid-19 spread', 'Doctors', 'Guidelines', 'Help Desk', 'Mask', 'Precautions', 'Report', 'Statistics', 'Symptoms', 'Types of mask', 'goodbye', 'greeting', 'hospital_search', 'noanswer', 'options', 'thanks', 'vaccination', 'vaccination_registeration']

############### DNN

training = []
output = []
labels = []
words = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net, 1)

model.load('./static/tf_model/model.tflearn')

###################

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method=='POST':
        query = request.form['content']
        result = make_reply(query)
    return render_template('index.html', result={query: query, result: result})

# @app.route('/', methods=('GET', 'POST'))
# def index():
#     if request.method=='POST':
#         content = request.form['content']
#         degree = request.form['degree']
#         todos.insert_one({'content': content, 'degree': degree})
#         return redirect(url_for('index'))

#     all_todos = todos.find()
#     return render_template('index.html', todos=all_todos)

# @app.post('/<id>/delete/')
# def delete(id):
#     todos.delete_one({"_id": ObjectId(id)})
#     return redirect(url_for('index'))

# @app.route('/webhook', methods = ['POST', 'GET'])
# def webhook():
# 	req = request.get_json(force = True)
# 	query = req['queryResult']['queryText']
# 	result = req['queryResult']['fulfillmentText']
# 	todos.insert_one({'query': query, 'result': result})
# 	print('Data inserted successfully')
# 	return Response(status=200)


###################

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def bot_initialize(user_msg):
    flag=True
    while(flag==True):
        user_response = user_msg
        if user_response.lower() == "quit" or user_response.lower() == "bye":
            break
        else:
          count=0
          bow = bag_of_words(user_response, words)
          results = model.predict(numpy.array([bow]))
          results_index = numpy.argmax(results)
          tag = labels[results_index]
        
          
          for tg in data["intents"]:
               if tg['tag'] == tag:
                      count=1
                      responses = tg['responses']
               
          if count == 1 :
                  bot_resp=random.choice(responses)
          else:
                  bot_resp="bye"

          return bot_resp

class telegram_bot():
    def __init__(self):
        self.token = "5398815839:AAEAVm4mpvAOYXoKfNO-WQHmtBQg9-xINS0"    #write your token here!
        self.url = f"https://api.telegram.org/bot{self.token}"
    def get_updates(self,offset=None):
        url = self.url+"/getUpdates?timeout=100"    # In 100 seconds if user input query then process that, use it as the read timeout from the server
        if offset:
            url = url+f"&offset={offset+1}"
        url_info = requests.get(url)
        return json.loads(url_info.content)
    def send_message(self,msg,chat_id):
        url = self.url + f"/sendMessage?chat_id={chat_id}&text={msg}"
        if msg is not None:
            requests.get(url)
    def grab_token(self):
        return tokens

tbot = telegram_bot()
update_id = None
def make_reply(msg):     # user input will go here
  
    if msg is not None:
        reply = bot_initialize(msg)     # user input will start processing to bot_initialize function
    return reply
       
while True:
    print("...")
    updates = tbot.get_updates(offset=update_id)
    # print(test_responses)
    updates = updates['result']
    print(updates)
    if updates:
        for item in updates:
            update_id = item["update_id"]
            print(update_id)
            try:
                message = item["message"]["text"]
                print(message)
            except:
                message = None
            from_ = item["message"]["from"]["id"]
            print(from_)
            reply = make_reply(message)
            tbot.send_message(reply,from_)

if __name__ == '__main__':
	app.run()
