from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import random
import pandas as pd 
import json
from sklearn.preprocessing import LabelEncoder

model = load_model('chatbot_model.keras')

with open('intents.json', 'r') as f:
    data = json.load(f)

dic = {"tag":[], "patterns":[], "responses":[]}
for example in data['intents']:
    for pattern in example['patterns']:
        dic['patterns'].append(pattern)
        dic['tag'].append(example['tag'])
        dic['responses'].append(example['responses'])

df = pd.DataFrame.from_dict(dic)

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])

lbl_enc = LabelEncoder() 
y = lbl_enc.fit_transform(df['tag'])

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

def generate(pattern): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
    x_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get")
def response():
    query = request.args.get('msg')
    res = generate(query)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)