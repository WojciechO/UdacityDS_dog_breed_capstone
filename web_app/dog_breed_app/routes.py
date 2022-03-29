from dog_breed_app import app

import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf

# Some utilites
import numpy as np
from dog_breed_app.util import base64_to_pil, face_detector, load_models, dog_detector, Resnet50_predict_breed


#loading models:
face_cascade, base_resnet, base_resnet_no_top, breed_prediction_model, dog_names  = load_models() 


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        is_human = face_detector(img, face_cascade)
        is_dog = dog_detector(img, base_resnet) 


        if not(is_human | is_dog):
            #Neither human face nor dog was recognised in the provided picture
            result = 'Neither human nor dog was recognised'
            pred_proba = 1.0
            # returning early, not to make unnecessary breed predictions
            return jsonify(result=result, probability=pred_proba)
        
        if (is_human | is_dog):
            predicted_breed = Resnet50_predict_breed(img, dog_names, base_resnet_no_top, breed_prediction_model)
            

        if is_human:
            print('Face detected')
            result = f'Recognised a human face in the picture. It mostly ressembles a {predicted_breed}'
            pred_proba = 1.0
        
        if is_dog:
            print('Dog detected')
            result = f'Recognised a dog in the picture. The predicted breed is {predicted_breed}'
            pred_proba = 1.0



        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None
