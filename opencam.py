# เปิดกล้องด้วย opencv
import cv2
import datetime
import face_recognition

cap = cv2.VideoCapture(0)

eyes_cascade=cv2.CascadeClassifier("python\python&opencv\haarcascade_eye_tree_eyeglasses.xml")
face_cascade=cv2.CascadeClassifier("python\python&opencv\haarcascade_frontalface_default.xml")

while(True):  
    check , frame =cap.read()  #รับภาพจากกล้อง frame ต่อ frame
    if check == True:

        gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#แปลงวีดีโอเป็นgray frame
        #จำแนกดวงตา
        scaleFactor =1.30
        minNeighber =5
        eyes_detect = eyes_cascade.detectMultiScale(gray_img,scaleFactor,minNeighber)
        face_detect = face_cascade.detectMultiScale(gray_img,scaleFactor,minNeighber)
        #แสดงตำแหน่งที่เจอใบหน้า
        for (x,y,w,h) in eyes_detect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=5)
            for (fx,fy,fw,fh) in face_detect:
                cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,0,0),thickness=5)
            
    currentdate=str(datetime.datetime.now())
    cv2.putText(frame,currentdate,(0,0),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,255,0),cv2.LINE_AA)
    cv2.imshow("output",frame)
    if cv2.waitKey(1) & 0xFF == ord("e"):

        break
cap.release()
cv2.destroyAllWindows()    