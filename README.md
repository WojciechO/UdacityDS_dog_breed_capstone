#Capston project for Udacity Data scientist nano degree

# Project overview
Presented project is the last step of a Data Science nano degree held by Udacity. 
  
Project consists of two parts:
* [jupyter notebook](notebook_files/dog_app.ipynb), which walks through the process of model trainig and creating final algorithm used in the app,
* a Flask web app [dog_breed app](dog_breed.py),
  * the app frontend is based on this [project](https://github.com/imfing/keras-flask-deploy-webapp)


# Objective and motivation
The primary objective of the project is recognising the dog breed based on the provided picture.  
Provided pictures go through the data processing and are then fed into separate models, which predict whether there is human or dog in the picture and what is the predicted dog breed. If human face is recognised, then model returns a dog breed that it ressembles the most.

I decided to work on this project, as it focuses on deep learning, which was not covered in the main course. Additionally, creating an app based on the created models touches upon skills that I was not very confident in, so it was a great opportunity to learn.

# Data

Data sets used for model training are not included in this repository, due to its big size.  
There were two datasets used to train and evaluate the performance of the model:
* Human faces images [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
* Dog images  [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

## Data processing
The models used in the algorithm require different image data processing:
* haarcascade face recognition algorithm requires image encoded in BGR colour scheme and then transformed to greyscale
* Tesnforflow models require the images are transformed to a (224,224) size and then transformed to a 4d tensor with shape of (1,224,224,3), where the final dimensions is used for the 3 colour channels. 

# Modelling overview

The final app algorithm consists of 3 models.
1. A face cascade model from open-cv recognises human faces on the pictures
2. A base ResNet50 model trained on whole Imagnet data can recognise 1000 classes of objects. This includes over 100 of dog breeds. The model is used to verify whether any dog is recognised in the picture.
3. A transfer learning was used to fine tune the base Res


## Metrics
Accuracy was used to assess the dog breed prediction model. There are 133 dog breeds in the training data set and final metric was calculated as average accuracy from all classes.  
The metric showed great improvement as we moved through different deep learning architecture:
* building a CNN from a scratch achieved ~3% of acuraccy
* Using transfer learning from VGG16 model by removing the final dense layer and training it on the dog pictures only, increased the accuracy to 45%
* Final approach used the transfer learning from ResNet50 model. After removing the final dense layer an additional Convoluted layer was added, followed by Global Average Pooling layer and the dense layer. Additionally droput and regularisation was included to tackle overfitting. The final model achieves average accuracy of 81% across all breeds.


# Application overview
The frontend of the app, especially the JS scripts and handling of the image transfer between the backend and frontend was inspired by this [project](https://github.com/imfing/keras-flask-deploy-webapp).
All licenses are retained as per the original repository.

Application uses the models described above to apply labels to the provided pictures.  
* If a human face is recognised then the most resembling dog breed is returned
![Alt text](screenshots/Example_app_human.png?raw=true)
* If dog is recognised, the dog breed is predicted
![Alt text](screenshots/Example_app_dog.png?raw=true)
* If neither dog nor human is detected, the dog breed prediction does not happen
![Alt text](screenshots/Example_app_horse.png?raw=true)



## Installation

Running the applications requires pipenv installed, as it is used as the packaging manager.

To run the application locally:
1. Install the requirements by runninng `pipenv install` in the repository root 
2. Uncomment the line 4 in dog_breed.py file
3. Next run `pipenv run python ./dog_breed.py`
4. App will start on http://0.0.0.0:3001/

## Running noteboook

* To run the notebooks make sure to run the `pipenv install --dev` command to include dependencies form the development environment
* To retrain the models, collect the datasets as described in the Data section


## Deployment

The app has been deployed to Heroku web app server, using pipenv as packaging tool and .
Additionally, the deployed applications required open-cv dependencies installed by apt. Those are included in the Aptfile and are installed on heroku upon deployment.

Unfortunatel  y, after deployment the app runs out of memory after loading in all required models. The free tier on heroku gives access to 512MB of RAM, which is not enough, given the selected architecture. First paid tier on heroku has the same memory allocation, only the more professional options go up to 1GB of RAM.  
A possible solutions to this problem include:
* changing the pretrained model from ResNet50 to a smaller model
* not using the base ResNet model to recognise dogs in the pictures

