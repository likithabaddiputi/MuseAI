import cv2
import os
import json

if not os.path.exists("dataset"):
    os.makedirs("dataset")

with open("C:/Users/Yamuna/OneDrive/jackfruit/images.json","r") as f:
    art_info=json.load(f)
dataset_histograms = {}
for i in art_info:
    img_path = i["file"]
    img = cv2.imread(img_path)
    if img is None:
        continue
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    cv2.normalize(hist, hist)
    dataset_histograms[img_path] ={ "hist":hist.flatten()# flatten makes a 2d array into 1d array
                                   ,"label": i["label"] }

count=0 # to give unique file name

cap = cv2.VideoCapture("https://192.168.0.103:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)#to reduce lag

while True:
    success,img =  cap.read()#cap.read()-- returns two parameters boolean value and frame
    if not success:
        print("not able to capture the image") 
        break
    display_frame =img.copy() # so that we dont modify the og
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    blur  = cv2.GaussianBlur(gray,(3,3),0)#odd numbers  
    edges=cv2.Canny(blur,50,150)#threshold 1 and 2 is the range using which image's edges are determined
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area=cv2.contourArea(c)
        if area >750:
            cv2.drawContours(display_frame,[c],-1,(0,255,0),2)
    cv2.imshow("Contours", display_frame)
    key=cv2.waitKey(1) & 0xFF
   
    hist_scan=cv2.calcHist([display_frame],[0],None,[256],[0,256])
    cv2.normalize(hist_scan, hist_scan)
    hist_scan_list=hist_scan.flatten()

    scores=[]
    for name, data in dataset_histograms.items():
        score = cv2.compareHist(hist_scan, data["hist"], cv2.HISTCMP_CORREL)
        scores.append((score,data["label"]))
    if scores:
        best_score,best_label=max(scores,key=lambda x :x[0])# max (returns a tuple)
        if best_score > 0.7:
            match_text=f"match : {best_label}"
        else:
            match_text="No Match"
    else:
        match_text="No References"
    cv2.putText(display_frame, match_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("result",display_frame)

    if key == ord('s'):
        filename=os.path.join("dataset",f"image_{count}.jpg")
        cv2.imwrite(filename,img)
        print(f"saved{filename}")
        count+=1
    elif key ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

