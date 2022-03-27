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

# Some utilites
import numpy as np
from dog_breed_app.util import base64_to_pil, face_detector, load_models, dog_detector



# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


#loading models:
face_cascade, base_resnet, breed_prediction_model  = load_models() 


# def model_predict(img, model):
#     img = img.resize((224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='tf')

#     preds = model.predict(x)
#     return preds


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
            # returning early, not to make unnecessary breed predictions, which require building of bottleneck features
            return jsonify(result=result, probability=pred_proba)
        
        if (is_human | is_dog):
            #TODO make prediction of the dog breed
            pass


        if is_human:
            print('Face detected')
            result = 'Human'
            #TODO: add dog breed predcition to the description
            pred_proba = 1.0
        
        if is_dog:
            print('Dog detected')
            result = 'Dog'
            #TODO: add dog breed predcition to the description
            pred_proba = 1.0

        
        # else:
        #     preds = model_predict(img, model)
        #     pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #     pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #     result = str(pred_class[0][0][1])               # Convert to string
        #     result = result.replace('_', ' ').capitalize()


        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None
