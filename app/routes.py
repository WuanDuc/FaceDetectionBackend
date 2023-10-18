from app import app
from flask import make_response, render_template, send_file
from flask import Flask, Response, request, jsonify
from io import BytesIO
import base64
import os
import sys
import cv2
import numpy as np
from base64 import b64decode
import imutils
import shutil
import time

def detect():
    #image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    path = 'image.jpeg'
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
    prototxt = 'deploy.prototxt'
    if os.path.isfile(prototxt):
        print("ok")
    else:
        print('[FILE NOT FOUND] prototxt is not found')
        return
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    if os.path.isfile(model):
        print("ok")
    else:
        print('[FILE NOT FOUND] model not found')
        return
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
            cv2.imwrite("image.jpeg", image)
def detect(image):
    #image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    image = imutils.resize(image, width=400)
    (h, w) = image.shape[:2]
    print(w,h)
    print("[INFO] loading model...")
    prototxt = 'deploy.prototxt'
    prototxt_age = 'age_deploy.prototxt'
    prototxt_gender = 'gender_deploy.prototxt'
    if (os.path.isfile(prototxt)):
        print("ok")
    else:
        print('[FILE NOT FOUND] prototxt is not found')
        return
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    model_age = 'age_net.caffemodel'
    model_gender = 'gender_net.caffemodel'
    if os.path.isfile(model):
        print("ok")
    else:
        print('[FILE NOT FOUND] model not found')
        return

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    net_age = cv2.dnn.readNetFromCaffe(prototxt_age, model_age)
    net_gender = cv2.dnn.readNetFromCaffe(prototxt_gender, model_gender)

    padding = 20

    image = imutils.resize(image, width=400)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
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
            # text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.putText(image, text, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            t = time.time()
            face = image[max(0,startY-padding):min(endY+padding,image.shape[0]-1),max(0,startX-padding):min(endX+padding, image.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            net_gender.setInput(blob)
            genderPreds = net_gender.forward()
            gender = genderList[genderPreds[0].argmax()]
            net_age.setInput(blob)
            agePreds = net_age.forward()
            age = ageList[agePreds[0].argmax()]

            label = "{},{}".format(gender, age)
            cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    return image
def detectImage():
    path = 'image.jpeg'
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
    output = detect(image)
    cv2.imwrite("image.jpeg", output)
def detectVideo():
  video_path = 'video.mp4'
  output_video_path = 'output_video.mp4'
  cap = cv2.VideoCapture(video_path)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  fps = int(cap.get(5))
  print(frame_width)
  print(frame_height)
  print(fps)


  ret, frame = cap.read()
  if not ret:
      return

  # Detect khuôn mặt trên frame
  frame = imutils.resize(frame, width=400)
  (h, w) = frame.shape[:2]

  out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mjpg'), fps, (w, h))
  image = detect(frame)
  # # Ghi frame đã detect vào video output
  out.write(image)
  while True:
      ret, frame = cap.read()
      if not ret:
          break

      # Detect khuôn mặt trên frame
      frame = imutils.resize(frame, width=400)
      (h, w) = frame.shape[:2]
      image = detect(frame)

      # # Ghi frame đã detect vào video output
      out.write(image)
  cv2.destroyAllWindows()
  cap.release()
  out.release()

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Wuan'}
    return render_template('index.html', title='Home', user=user)
@app.route('/image', methods=['GET', 'POST'])
def image():
    if(request.method == "POST"):
        bytesOfImage = request.get_data()
        with open('image.jpeg', 'wb') as out:
            #out.write(bytesOfImage)
            out.write(base64.decodebytes(bytesOfImage))
        detectImage()
        with open('image.jpeg', 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
        #response = make_response(send_file(file_path,mimetype='image/png'))
        response = make_response(encoded_string)
        response.headers['Content-Transfer-Encoding']='base64'
        print(response)
        return response 
@app.route("/video", methods=['GET', 'POST'])
def video():
    if(request.method == "POST"):
        bytesOfVideo = request.get_data()
        with open('video.mp4', 'wb') as out:
            out.write(base64.b64decode(bytesOfVideo))
        detectVideo()
        with open("al.mp4", "rb") as videoFile:
            text = base64.b64encode(videoFile.read())
            print(text)
        response = make_response(text)
        response.headers['Content-Transfer-Encoding']='base64'
        return response

