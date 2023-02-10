import flask
import io
import string
import time
import os
import numpy as np
import os
from numpy.core.fromnumeric import argmax

from numpy.lib.function_base import meshgrid
import tensorflow as tf
from PIL.Image import core as _imaging
from PIL import Image
from flask import Flask, jsonify, request

PROJECT_ROOT = os.getcwd()

MODEL_LEAF_NO_LEAF_PATH = os.path.join(PROJECT_ROOT, 'models', 'leaf_no_leaf')
MODEL_ADDR_4_CLASS_PATH = os.path.join(PROJECT_ROOT, 'models', 'addr_4_class')


modelLeafNoLeaf = tf.keras.models.load_model(MODEL_LEAF_NO_LEAF_PATH)
modelAddr4Class = tf.keras.models.load_model(MODEL_ADDR_4_CLASS_PATH)

classes_leaf_no_leaf = ['Apple Leaf', 'Not Apple Leaf']
apple_4_classes = ['BlackRot', 'Healthy', 'Rust', 'Scab']


def prepare_image(img, width, height):
    dims = (width, height)
    img = Image.open(io.BytesIO(img))
    img = img.resize(dims)
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def isImgAnAppleLeaf(processed_img):
    pred = modelLeafNoLeaf.predict(processed_img)
    if(np.argmax(pred) == 0):
        return True
    else:
        return False


def predict_result(img):
    pred = modelLeafNoLeaf.predict(img)
    classes = ['Apple Leaf', 'Not Apple Leaf']
    return classes[np.argmax(pred)]


def predictOnImage(img_full):
    return modelAddr4Class.predict(img_full)


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    # basic validation
    if 'file' not in request.files:
        return jsonify(status=False, data={'label': '', 'class': -1, 'confidence': -1}, message="Please add an image")
    file = request.files.get('file')
    if not file:
        return jsonify(status=False, data={'label': '', 'class': -1, 'confidence': -1}, message="Image Not Valid")

    img_bytes = file.read()
    # resize img to 64,64 as binary model supports that size
    img_small = prepare_image(img_bytes, 64, 64)

    # check if img is a leaf or not
    # if it is leaf the proceed with actual classification
    # else return with msg

    if(not isImgAnAppleLeaf(img_small)):
        return jsonify(status=False, data={'label': '', 'class': -1, 'confidence': -1}, message="Image is not a leaf")
    else:
        img_full = prepare_image(img_bytes, 224, 224)
        pred = predictOnImage(img_full)
        predicted_class = int(np.argmax(pred))
        label = apple_4_classes[predicted_class]
        confidence = float(pred[0][predicted_class])
        return jsonify(status=True, data={'label': label, 'class': predicted_class, 'confidence': confidence}, message="Prediction Success")


@app.route('/')
def index():
    return jsonify(status=True, data=modelLeafNoLeaf.summary(), message="hello")
