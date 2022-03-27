from dog_breed_app import app

import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from .util import base64_to_pil, face_detector, load_models



# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


#loading models:
face_cascade, base_resnet, breed_prediction_model  = load_models() 


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        has_face = face_detector(img, face_cascade)

        if has_face:
            print('Face detected')
            result = 'Human'
            pred_proba = 1.0
        # Make prediction
        
        else:
            preds = model_predict(img, model)
            pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
            pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
            result = str(pred_class[0][0][1])               # Convert to string
            result = result.replace('_', ' ').capitalize()


        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None
