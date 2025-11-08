import cv2

cap = cv2.VideoCapture("https://100.99.230.5:8080/video")

while True:
    success,img =  cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    blur  = cv2.GaussianBlur(gray,(3,3),0)
    cv2.imshow("edges",cv2.Canny(blur,50,50))

cap.release()

