

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model




# Flask utils
from flask import Flask
from flask_ngrok import run_with_ngrok



from flask import Flask, redirect, url_for, request, render_template
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2

# Define a flask app
app = Flask(__name__)



# Model saved with Keras model.save()
MODEL_PATH = 'CNN.h5'


# Load your trained model
model = load_model(MODEL_PATH)
          # Necessary
model.compile(loss='categorical_crossentropy',

              optimizer='Adam',metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

#You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
import os


#model.save('models/model_resnet.h5')

print('Model loaded. Check http://127.0.0.1:5000/')

from PIL import Image


def model_predict(x,model):
    #img = image.load_img(img_path, target_size=(224, 224))

    #Preprocessing the image
    #x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction

    try:
      preds = model.predict(x)
      print(preds)
      clas=np.argmax(preds)
      print(clas)
      if clas==0:
        return "Tuber"
      elif clas==1:
        return "Normal"
    except:
      return "Image is Incompatible"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data =request.files['file']
        img = Image.open(request.files['file'])

        #image preprocessing
        img = np.array(img)
        img = cv2.resize(img,(128,128))
        img = np.expand_dims(img, axis=0)

        #img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
      

        '''f = request.files['file']

        filename = secure_filename(f.filename) # save file 
        filepath = os.path.join(app.config['imgdir'], filename);

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(
            #basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)'''

        # Make prediction
        preds = model_predict(img,model)

        # for ResNet
        #Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run()