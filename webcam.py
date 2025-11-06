import cv2

count=0 # to give unique file name

cap = cv2.VideoCapture("https://192.168.0.103:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)#to reduce lag
while True:
    success,img =  cap.read()#cap.read()-- returns two parameters boolean value and frame 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    blur  = cv2.GaussianBlur(gray,(3,3),0)#odd numbers  
    cv2.imshow("edges",cv2.Canny(blur,50,50))
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blur, contours, -1, (255,0,0), 2)
    for c in contours:
        area=cv2.contourArea(c)
        if area >750:
            cv2.drawContours(img,[c],-1,(255,0,0),2)
    cv2.imshow("Contours", img)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename=f"image_{count}.jpg"
        cv2.imwrite(filename,img)
        print(f" saved{filename}")
        count+=1
    elif key ==ord('q'):
        break

cap.release()

    