
from types import GeneratorType
import cv2
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

#img = cv2.imread('RDJ.jpg')
#img = cv2.imread('jp.jpg')
webcam = cv2.VideoCapture(0)

while True :
  Successful_fram_read, frame = webcam.read()

  Grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  faces = trained_face_data.detectMultiScale(Grayscaled_img)

  for (x, y, w, h) in faces: 
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

  cv2.imshow('Clever programmer Raf',frame)
  cv2.waitKey(1)
 
  #print("compelet")

""
#Detect faces w `1`

#faces = trained_face_data.detectMultiScale(Grayscaled_img)

#for (x, y, w, h) in faces:
  #cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)

#print(faces)

#cv2.imshow('Clever programmer Raf',img)

#cv2.waitKey(1)
""
