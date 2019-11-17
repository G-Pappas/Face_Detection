"""
This script allows you to detect faces on pictures.
In order this script to work picture must be in the same folder with the script, also the picture must be named "picture.jpg".
Cascade function used for frontal faces, for highter accuracy more functions musted be used.
"""
import cv2

#load cascade file in python
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#import image
img = cv2.imread("picture.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert image to grey scale for highter accuracy
                                                #I use a second variable so I can keep original picture

#detect the coordinates of the face and get values into a numpy array ex: 157  84 379 379
faces = face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1,# Tell python how much to reduse scale to search for smaller faces, low number high accuracy but slow speed
minNeighbors=5) # Higher minNeighbors might cause less detections but with higher quality, if it is low there might be false returns


#highlight the face
for x, y, w, h in faces:                            #3 is the width of the rectangle line
    img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3) #I need to give the top left start point and bottom right point
                                           #0,255,0 is color in bgr

print(faces)# this prints out the coordinates of the face


cv2.imshow("Img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
