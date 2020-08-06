import numpy as np
import string
import pickle as pkl
from flask import Flask,request,render_template
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(text):
    prepro_text=str(text).lower().replace('-',' ')
    translation_table=str.maketrans('\n',' ',string.punctuation+string.digits)
    prepro_text=prepro_text.translate(translation_table)
    return prepro_text

model=pkl.load(open('model.pkl', 'rb'))
vectorizer=pkl.load(open('vectorizer.pkl', 'rb'))
app=Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method=='POST':
		text=request.form['Input Language Script']
		text=preprocess(text)
		data=[text]
		vect=vectorizer.transform(data).toarray()
		lang=model.predict(vect)
	return render_template('index.html', prediction_text='The written language script is {}'.format(lang))


if __name__=='__main__':
	app.run(debug=True)
