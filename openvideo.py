# เปิดวีดีโอด้วย opencv
import cv2
import datetime
import numpy as np

# classes=["omen","wukong","raz","grakk","joker","airi","laville","lumburr","nakroth","liliana"]
# color=np.random.uniform(0,100,size=(len(classes),3))
cap= cv2.VideoCapture("python/python&opencv/rov_clip.mp4")

while(cap.isOpened()):  
    check , frame =cap.read()  #รับภาพจากกล้อง frame ต่อ frame

    if check == True:
        currentdate=str(datetime.datetime.now())
        cv2.putText(frame,currentdate,(40,50),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,255,0),cv2.LINE_AA)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#แปลงวีดีโอเป็นgray frame
        cv2.imshow("output",frame)
        if cv2.waitKey(100) & 0xFF == ord("e"):
            break
    else :
        break
cap.release()
cv2.destroyAllWindows()    