
import cv2
#อ่านไฟล์สำหรับการแยกแยะประเภทใบหน้า
face_cascade=cv2.CascadeClassifier("open_cam_recognition\haarcascade_frontalface_default.xml")

# สร้าง dataset
def create_dataset(img,id,img_id):
    cv2.imwrite("open_cam_recognition\data\pic"+str(id)+"."+str(img_id)+".jpg",img)

# สร้างกรอบที่จะทำการ detection
def draw_boundary(img,classifier,scaleFactor,minNeighbers,color,clf):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#แปลงวีดีโอเป็นgray frame
    face_detect =classifier.detectMultiScale(gray,scaleFactor,minNeighbers)
    coords=[]
    #แสดงตำแหน่งที่เจอใบหน้า
    for (x,y,w,h) in face_detect:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        id,con=clf.predict(gray[y:y+h,x:x+w])
        if con<=100:
            cv2.putText(img,"kiddy",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2) 
        else:
            cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        if (con<=100):    
            con="  {0}%".format(round(100-con))
        else:
            con="  {0}%".format(round(100-con))
        print(str(con))
        
        # -------------------------------------------
        # ใช้ idระบุว่าเป็นใคร id1คือหน้าkiddy
        # if id==1:
        #     cv2.putText(img,"kiddy",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        # -----------------------------------
        coords=[x,y,w,h]
    return img,coords
def detect(img,face_cascade,img_id,clf):
    img,coords=draw_boundary(img,face_cascade,1.1,10,(0,0,255),clf)
    if len(coords)== 4:
        id=1
        result = img[coords[1]: coords[1]+coords[3],coords[0]: coords[0]+coords[2]]
        # create_dataset(result,id,img_id)
    return img

img_id=0

cap= cv2.VideoCapture(0)
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier2.xml")
while(True):
    ret,frame=cap.read()
    frame=detect(frame,face_cascade,img_id,clf)
    cv2.imshow("frame",frame)
    img_id+=1
    if(cv2.waitKey(1) and 0xFF==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()    


