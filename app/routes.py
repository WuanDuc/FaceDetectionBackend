from app import app
from flask import render_template, send_file
from flask import Flask, Response, request, jsonify
from io import BytesIO
import base64
import os
import sys
import cv2
import numpy as np
from base64 import b64decode
import imutils

def detect():
    #image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    path = '../image.jpeg'
    if os.path.isfile(path):
        print("ok")
    else:
        print('FILE NOT FOUND')
        return
    image = cv2.imread(path)
    assert not isinstance(image,type(None)), 'image not found'
    if image is None:
        print('Wrong path:', path)
    else:
        print('[RIGHT] right path:', path)
        print('[INFO] imagesize:', image.shape)
    image = imutils.resize(image, width=400)
    (h, w) = image.shape[:2]
    print(w,h)
    print("[INFO] loading model...")
    prototxt = '../deploy.prototxt'
    model = '../res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    image = imutils.resize(image, width=400)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX,startY,endX,endY)
            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imwrite("../image.jpeg", image)


@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Wuan'}
    return render_template('index.html', title='Home', user=user)
@app.route("/image", methods=['GET', 'POST'])
def image():
    if(request.method == "POST"):
        bytesOfImage = request.get_data()
        with open('../image.jpeg', 'wb') as out:
            #out.write(bytesOfImage)
            out.write(base64.decodebytes(bytesOfImage))
        detect()
        return "Image read"
@app.route("/video", methods=['GET', 'POST'])
def video():
    if(request.method == "POST"):
        bytesOfVideo = request.get_data()
        with open('../video.mp4', 'wb') as out:
            out.write(bytesOfVideo)
        return "Video read"
@app.route("/get_image", methods=['GET'])
def get_image():
    image_path = 'image.jpeg'

    try:
        return send_file(image_path, mimetype='image/jpeg')
    except FileNotFoundError:
        return "Image not found", 404