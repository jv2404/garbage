import os
import numpy as np
from keras. layers import Dense, Flatten
from keras. models import Model
from keras .applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing. image import ImageDataGenerator , load_img , img_to_array
import keras

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "images"
STATIC_FOLDER = "static"

# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "best_model.h5")


# Preprocess an image
# home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    imagefile=request.files['image']
    image_path = "./images/"+ imagefile.filename
    imagefile.save(image_path)

    imgage = load_img(image_path, target_size=(256,256) )
    i = img_to_array(imgage)
    i = preprocess_input(i)
    input_arr = np.array([i])
    input_arr.shape
    pred = np.argmax(cnn_model .predict(input_arr))
    if pred ==0:
       label= ("The. image is of garbage ")
    else:
        label=("The image is of potholes ")

    
    return render_template(
        "classify.html",image_file_name=imagefile.filename,  label=label
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True
