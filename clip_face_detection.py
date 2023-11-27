
import cv2
import numpy 
#อ่านvideo
cap= cv2.VideoCapture("python\python&opencv\X2Download.app-MOST HANDSOME NBA PLAYERS 2020 feat Kyle Kuzma(360p).mp4")

#อ่านไฟล์สำหรับการแยกแยะประเภทใบหน้า
face_cascade=cv2.CascadeClassifier("python\python&opencv\haarcascade_frontalface_default.xml")

while(cap.isOpened()):  
    check , frame =cap.read()  #รับภาพจากกล้อง frame ต่อ frame

    if check == True:

        gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#แปลงวีดีโอเป็นgray frame
        #จำแนกใบหน้า
        scaleFactor =1.30
        minNeighber =5
        face_detect = face_cascade.detectMultiScale(gray_img,scaleFactor,minNeighber)
        #แสดงตำแหน่งที่เจอใบหน้า
        for (x,y,w,h) in face_detect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=5)
            cv2.imshow("output",frame)
        if cv2.waitKey(10) & 0xFF == ord("e"):
            break
    else :
        break
cap.release()
cv2.destroyAllWindows()    


