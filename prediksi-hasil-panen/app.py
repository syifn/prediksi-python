from logging.handlers import BaseRotatingHandler
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import random

app = Flask(__name__, template_folder='template')

def load_model():
    with open('model_dtr.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form-prediksi')
def form_prediksi():
    return render_template('formpage.html')

@app.route('/prediksi-hasil-panen', methods=['POST'])
def predict():
    
    LuasLahan = float(request.form['LuasLahan'])
    JumlahBibit = float(request.form['JumlahBibit'])
    JumlahPakan = float(request.form['JumlahPakan'])

    hitung = np.array([[LuasLahan,JumlahBibit,JumlahPakan]])
    
    prediction = model.predict(hitung)
    hasil_pred = round(prediction[0], 2)
    
    return render_template('hasil.html', hasil=hasil_pred, luas=LuasLahan, jmlB=JumlahBibit, jmlP=JumlahPakan)

if __name__ == '__main__':
    app.run(debug = True)
