"""Utilities
"""
import re
import base64

import numpy as np

import os

from PIL import Image
from io import BytesIO

import cv2  
from keras.models import load_model
from keras.applications.resnet import ResNet50


# image processing
def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

def pil_to_grey(PIL_image):
    '''Converts PIL Image to a  cv2 GRAY representation'''
    img_bgr = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def load_models():
    ''' Function returning all model objects used in the algorithm
    RETURNS:
    face_cascade -- Cascade face recognition model
    base_resnet -- Base Resnet model, trained on IMagenet
    breed_prediction_model -- Model predicting dog breeds
    '''
    package_directory = os.path.dirname(os.path.abspath(__file__))
    filename_haars_cascade = os.path.join(package_directory, './models/haarcascade_frontalface_alt.xml')
    filename_breed_pred = os.path.join(package_directory, './models/resnet_breed_prediction.hd')

    print(package_directory)

    face_cascade = cv2.CascadeClassifier(filename_haars_cascade)
    #cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print('Loaded face cascade model')

    # In local development, the base Resnet can be stored locally, to speed up the loading
    #base_resnet = load_model('./models/base_resnet.hd')
    base_resnet = ResNet50(weights='imagenet')
    base_resnet.make_predict_function()
    print('Loaded base Resnet model')

    breed_prediction_model = load_model(filename_breed_pred)
    breed_prediction_model.make_predict_function()
    print('Loaded breed reconigtion model')

    return face_cascade, base_resnet , breed_prediction_model

def face_detector(img, face_detection_model):
    ''' Function using provided face detection model, returning true when at leat one human face was detected in the picture'''
    
    gray_img = pil_to_grey(img)
    faces = face_detection_model.detectMultiScale(gray_img)

    return len(faces) > 0


