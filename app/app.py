import os
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras.backend as K
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.python.framework import ops

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'

def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    # load the model we saved
    model = load_model(MODEL_FOLDER+'/model.h5')
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    print('[INFO] : Model loaded')
    return model

def predict(fullpath):
# predicting images
    model=load__model()
    Image = cv2.imread(fullpath)
    Image = cv2.resize(Image, (96, 96))
    image_array = img_to_array(Image)  
    x= np.array(image_array,dtype='float')/255.0
    if(K.image_data_format() == 'channels_first'):
        x = x.reshape(1,3,96,96)
    else:
        x = x.reshape(1,96,96,3)
    images = np.vstack([x])
    result= model.predict(images)
    return result
# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)
        result = predict(fullname)
        
        if result > .5:
            label = 'Abnormal'
            accuracy = result * 100
        else:
            label = 'Normal'
            accuracy = (1-result) * 100
        
        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
