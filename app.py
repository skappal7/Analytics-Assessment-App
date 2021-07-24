pip install flask

pip install pycaret

from flask import Flask, url_for, redirect, render_template, jsonify
from pycaret.classification import*
import pandas as pd
import numpy as np

app = Flask(__name__)

model = load_model('Final RF Model 23JUL2021')
cols = ['AHT','NTT','Sentiment','Complaints','Repeats']

@app.route('/')
def home():
	return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
	int_features = [x for x in request.form.values()]
	final = np.array(int_features)
	data_unseen = pd.DataFrame([finak], columns = cols)
	prediction = predict_model(model, data=data_unseen, round=0)
	prediction = int(prediction.Label[0])
	return render_template('home.html',pred='Predicted Maturiy Level is{}'.format(prediction))
