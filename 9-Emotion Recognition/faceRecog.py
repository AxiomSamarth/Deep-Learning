#!/usr/bin/env python

import numpy as np
import cv2
from emotions import find_emotion

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#cap = cv2.VideoCapture(0);
rec = cv2.face.createLBPHFaceRecognizer();
rec.load("trainner/trainningData.yml")
id=0;
#font=cv2.InitFont(cv2.FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4);

#cv2.COLOR_BGR2GRAY

images = ['panda.png', 'family.jpg', 'me_sisters.jpg', 'iitrpr.jpg', 'me_anusha_akka.jpg', 'sad.jpg', 'iisc.jpg']

for image in images:
    img = cv2.imread('Test/' + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi = cv2.resize(gray[y:y+h,x:x+w], (48,48))
        result = find_emotion(roi)
        '''if(id==1):
            id="SAGAR"
        elif(id==2):
            id="VIGNESH"
        elif(id==3):
            id="RAJKUMAR"
        elif(id==4):
            id="Girish"
        else:
            id="UNKNOWN"
        #cv2.putText(img,str(id),(x,y+h),font,255);'''
        cv2.putText(img, result, (x,y+h),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2);
    cv2.imshow('Emotion_Recognition',img);
    cv2.waitKey(0)
    #cap.release()
    cv2.destroyAllWindows()
			

			
