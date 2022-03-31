"""Utilities
"""
import re
import base64
import pickle

import numpy as np

import os

from PIL import Image
from io import BytesIO

import cv2  
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet
from keras.preprocessing import image

import tensorflow as tf

from keras.applications.imagenet_utils import preprocess_input



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

def pil_to_tensor(PIL_image):
    '''Converts PIL Image to a 4d tensot, accrepted by TF models '''
    img = PIL_image.resize((224, 224))
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

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
    base_resnet_no_top - Base Renset model without the final dense layer
    breed_prediction_model -- Model predicting dog breeds
    dog_names - array with dog breeds
    '''
    package_directory = os.path.dirname(os.path.abspath(__file__))
    filename_haars_cascade = os.path.join(package_directory, './models/haarcascade_frontalface_alt.xml')
    filename_breed_pred = os.path.join(package_directory, './models/resnet_breed_prediction.hd')
    #filename_base_resnet_no_top = os.path.join(package_directory, './models/base_resnet_no_top.hd')
    filename_dog_names = os.path.join(package_directory, './models/dog_labels.pkl')

    face_cascade = cv2.CascadeClassifier(filename_haars_cascade)
    #cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print('Loaded face cascade model')

    # In local development, the base Resnet can be stored locally, to speed up the loading
    #base_resnet = load_model('./models/base_resnet.hd')
    base_resnet = ResNet50(weights='imagenet')
    base_resnet._make_predict_function()
    print('Loaded base Resnet model')

    base_resnet_no_top = ResNet50(weights='imagenet', include_top=False)
    base_resnet_no_top._make_predict_function()
    # base_resnet_no_top = load_model(filename_base_resnet_no_top)
    print('Loaded base Resnet model without last dense layer')

    breed_prediction_model = load_model(filename_breed_pred)
    breed_prediction_model._make_predict_function()
    
    #required workardound for issues with TF graph, which occur with TF < 2.0
    global model_graph
    model_graph = tf.get_default_graph() 
    print('Loaded breed reconigtion model')

    with open(filename_dog_names, 'rb') as f:
        dog_names = pickle.load(f)


    return face_cascade, base_resnet , base_resnet_no_top, breed_prediction_model, dog_names

def face_detector(img, face_detection_model):
    ''' Function using provided face detection model, returning true when at leat one human face was detected in the picture'''
    
    gray_img = pil_to_grey(img)
    faces = face_detection_model.detectMultiScale(gray_img)

    return len(faces) > 0


def ResNet50_predict_labels(PIL_img, resnet_model):
    ''' returns prediction vector for provided PIL image '''
    img = preprocess_input(pil_to_tensor(PIL_img))
    return np.argmax(resnet_model.predict(img))

def dog_detector(PIL_img, resnet_model):
    '''Function verifying if there is dog in the provided picture. 
    Based on the base Resnet50 model, trained on the whole Imagenet.
    Dog classes in the Imagenet have labels between 151 and 268 
    '''
    prediction = ResNet50_predict_labels(PIL_img, resnet_model)
    return ((prediction <= 268) & (prediction >= 151)) 


def Resnet50_predict_breed(PIL_img, dog_names, base_resnet_no_top, resnet_model):
    '''Function returning dog breed prediction.
    Model uses transfer learning from base Resent, so input needs to be transformed to the nottleneck features
    INPUT:
    PIL_img: image in PIL format
    dog_names: array with dog breeds from the Imagenet
    base_resnet_no_top: Base Resnet50 model without the final dense layer
    resnet_model: Final ResNet model, which predicts the dog breed

    RETURNS:
    predicted dog breed
    '''
    img = preprocess_input_resnet(pil_to_tensor(PIL_img))

    # extract bottleneck features
    bottleneck_feature = base_resnet_no_top.predict(img)
    print('bottle neck obtained')
    print(bottleneck_feature.shape)
    # obtain predicted vector
    resnet_model._make_predict_function()

    #required workardound for issues with TF graph, which occur with TF < 2.0
    with model_graph.as_default():
        predicted_vector = resnet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]




