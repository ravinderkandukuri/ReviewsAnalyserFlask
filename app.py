import pickle

import nltk
from flask import Flask, render_template, request


# load the model from disk

clf = pickle.load(open('nlp_modelnew.pkl', 'rb'))
cv = pickle.load(open('tranformnew.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():



    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction)
    return render_template('home.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
