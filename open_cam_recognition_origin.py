import cv2
# สร้าง datasetimport cv2
#อ่านไฟล์สำหรับการแยกแยะประเภทใบหน้า
face_cascade=cv2.CascadeClassifier("open_cam_recognition\haarcascade_frontalface_default.xml")
def create_dataset(img,id,img_id):
    cv2.imwrite("open_cam_recognition\data\pic"+str(id)+"."+str(img_id)+".jpg",img)

# สร้างกรอบที่จะทำการ detection
def draw_boundary(img,classifier,scaleFactor,minNeighbers,color,text):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#แปลงวีดีโอเป็นgray frame
    face_detect =classifier.detectMultiScale(gray,scaleFactor,minNeighbers)
    coords=[]
    #แสดงตำแหน่งที่เจอใบหน้า
    for (x,y,w,h) in face_detect:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        coords=[x,y,w,h]
    return img,coords
def detect(img,face_cascade,img_id):
    img,coords=draw_boundary(img,face_cascade,1.1,10,(0,0,255),"Face")
    id=1
    if len(coords)== 4:
        result = img[coords[1]: coords[1]+coords[3],coords[0]: coords[0]+coords[2]]
        create_dataset(result,id,img_id)
    return img

img_id=0

#อ่านvideo
cap= cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    frame=detect(frame,face_cascade,img_id)
    cv2.imshow("frame",frame)
    img_id+=1
    if(cv2.waitKey(1) and 0xFF==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()    


