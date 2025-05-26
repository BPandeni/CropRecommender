from flask import Flask, render_template, request
import numpy as np
import pandas
import sklearn
import pickle
from waitress import serve

model = pickle.load(open('modelrfczim.pkl', 'rb'))
with open('label_encoder_zim.pkl', 'rb') as le:
    label_encoder = pickle.load(le)

app = Flask(__name__)

@app.route('/')
def signup():
    return render_template('signup.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity =  float(request.form['Humidity'])
        ph =  float(request.form['pH'])
        average_rainfall = float(request.form['Rainfall_Amount'])
        agroecological_region = request.form['Agroecological_region']
        
        encoded_region = label_encoder.transform([agroecological_region])[0]
        single_pred = np.array([N,P,K,temperature,humidity,ph,average_rainfall,encoded_region]).reshape(1, -1)
        
        prediction = model.predict(single_pred)

        crop_images = {
            'wheat':'wheat.jpeg',
            'maize': 'maize.jpeg',
            'cotton': 'cotton.jpeg',
            'millet': 'millet.jpeg',
            'tobacco': 'tobacco.jpeg',
            'sugarcane': 'sugarcane.jpeg',
            'sorghum': 'sorghum.jpeg',
            'coffee': 'coffee.jpeg'
        }
        
        if prediction and prediction[0]:
            crop = prediction[0]
            result = f'{crop} is the best crop to grow'
            crop_image = crop_images.get(crop, 'crop.jpeg')
        else:
            f'could not determine the right crop to grow'
            crop_image = 'crop.jpeg'
    
    except ValueError as e:
        result = f'Valuer error: {str(e)}'
        crop_image = 'error.jpeg'
    except TypeError as e:
        result = f'Type Error: {str(e)}'
        crop_image = 'error.jpeg'
    except Exception as e:
        result = f"An error occurred: {str(e)}"
        crop_image = 'error.jpeg'


    return render_template('index.html', result = result, crop_image = crop_image )


if __name__ == '__main__':
    app.run(debug = False)
    #serve(app, host = '0.0.0.0', port = 5000)
