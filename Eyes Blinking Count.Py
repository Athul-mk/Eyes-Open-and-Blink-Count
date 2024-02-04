import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier(r'C:\Users\ATHUL AKSHAY\Downloads\haarcascade_frontalface_default(1).xml')
eye_cascade=cv2.CascadeClassifier(r'C:\Users\ATHUL AKSHAY\Downloads\haarcascade_eye(1).xml')

first=True
cap=cv2.VideoCapture(0)
ret,image=cap.read()
count=0

while ret:
    ret,image=cap.read()
    gc = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image,1.2,4)
    if len(faces)>0:
        for (x,y,w,h) in faces:
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            face_detector=gc[y:y+h, x:x+w]
            face_clr=image[y:y+h, x:x+w]

            eyes=eye_cascade.detectMultiScale(face_detector,1.2,4)
            if len(eyes)>=2:
                if first:
                    cv2.putText(image,'Eyes Detected',(70,70),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
                else:
                    cv2.putText(image,'Eyes Open',(70,70),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
            else:
                if not first:
                    cv2.putText(image,'Eyes Blinked',(70,70),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
                    count+=1
                    print('Eyes blinked',count)

        first=False
    else:
        first=True
        cv2.putText(image,'No Face Detected',(70, 70),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)

    cv2.imshow('Blink',image)
    a=cv2.waitKey(1)

    if a==ord('e'):
        break

cap.release()
cv2.destroyAllWindows()