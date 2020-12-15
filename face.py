import cv2
import numpy as np
import face_recognition
import os

path= 'image'
images= []
classname = []
ilist= os.listdir(path)
#print(ilist)


for cl in ilist:
    curimage=cv2.imread(f'{path}/{cl}')
    images.append(curimage)
    classname.append(os.path.splitext(cl)[0])
#print(classname)

def findencoding(images):
    encoded=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encoded.append(encode)
    return encoded
    
    
from datetime import datetime
def markattendance(name):
    with open('att.txt','r+') as f:
        mydatalist = f.readlines()
        namelist =[]
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        #print(mydatalist)

#print('encodedlist completed')


cap= cv2.VideoCapture(0)

while True:
    sucess,img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    
    faceloc = face_recognition.face_locations(imgs)
    encode= face_recognition.face_encodings(imgs, faceloc)
    
    for encode,faceloc in zip(encode, faceloc):
        matches = face_recognition.compare_faces(encodedlist,encode)
        facedis = face_recognition.face_distance(encodedlist,encode)
        #print(facedis)
        matchindex = np.argmin(facedis)
        
        if matches[matchindex]:
            name = classname[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            markattendance(name)
            
            
            
    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows(
