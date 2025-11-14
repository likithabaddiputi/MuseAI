import streamlit as st
import cv2
import os
import json

st.markdown("<h1 style='text-align: center;'>MUSE AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Click the button below to scan the object.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,1,2])
with col2:
    start = st.button("Start webcam")

if start:    # Ensure dataset directory exists in the working folder
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    # Loads art information from images.json
    with open("images.json","r") as f:
        art_info=json.load(f)

    dataset_histograms = {}

    for i in art_info: #iterate through each art piece info, one dictionary at a time
        img_path = i["file"] #gets the image file path
        img = cv2.imread(img_path)  #reads the image from the file path
        if img is None:    #if image is not read properly, skip to next
            continue
        hist = cv2.calcHist([img], [0], None, [256], [0,256])   #Tells us how many pixels are there for each intensity value from 0-255. {gray scale image [0]}
        cv2.normalize(hist, hist) #Normalizes the histogram values in place
        dataset_histograms[img_path] ={ "hist":hist.flatten()# flatten makes a 2d array into 1d array
                                   ,"label": i["label"], "backstory": i["backstory"], "audio": i["audio"] }

    count=0 # to give unique file name

    cap = cv2.VideoCapture(0) #use 0 for webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)#to reduce lag

    info_box = st.empty()
    audio_box = st.empty()

    if "last_label" not in st.session_state:
        st.session_state.last_label = None


    while True:
        success,img =  cap.read()#cap.read()-- returns two parameters boolean value and frame
        if not success:
            print("not able to capture the image") 
            break
        display_frame =img.copy() # so that we dont modify the og
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray,(3,3),0)#odd numbers  #blurs the image to reduce noise and details 
                                            #(3,3) is the kernel size meaning kernel is the collection of pixels. first index represents width and second represents height
                                            # kernel contains the weight of the pixel when we process the image

        edges=cv2.Canny(blur,50,150)#threshold 1 and 2 is the range using which image's edges are determined
                                #The change in brightness is calculated and if it is within the threshold values, it is considered as an edge 
                                #the values below 50 are considered as non-edges and above 150 as strong edges
                                #edges are where there is a sharp change in brightness

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         # contours is a list of all points that make up the edges
        #hierarchy is the relationship between contours like what are inner contours and outer contours 

        for c in contours:
            area=cv2.contourArea(c) #The area enclosed within the contour points
            if area >750:
                cv2.drawContours(display_frame,[c],-1,(0,255,0),2) #-1 means draw all contours, (0,255,0) is the color green, 2 is the thickness

        key=cv2.waitKey(1) & 0xFF
   
        hist_scan=cv2.calcHist([display_frame],[0],None,[256],[0,256])
        cv2.normalize(hist_scan, hist_scan)
        hist_scan_list=hist_scan.flatten() #hist_scan is 2d array, flatten makes it 1d array

        scores=[]
        for name, data in dataset_histograms.items():
            score = cv2.compareHist(hist_scan, data["hist"], cv2.HISTCMP_CORREL) #compares two histograms using correlation method 
                                                                             # 1 for perfect match and -1 for no match
            scores.append((score,data["label"]))
        if scores:
            best_score,best_label=max(scores,key=lambda x :x[0])#key=lambda x :x[0] means we are comparing based on the first element of the tuple which is score
                                                             #lambda is an anonymous function that takes x as input and returns x[0]
        if best_score > 0.7:
            if best_score > 0.7:
                match_text=f"match : {best_label}"
            else:
                match_text="No Match"
        else:
            match_text="No References"

# ---- PUT TEXT + AUDIO INTEGRATION HERE ----
        if best_score > 0.7:
        # Find the matched object's details
            for img_path, data in dataset_histograms.items():
                if data["label"] == best_label:
                    matched_backstory = data["backstory"]
                    matched_audio = data["audio"]
                    break

    # Show details in Streamlit
            if st.session_state.last_label != best_label:
                info_box.markdown(f"""
            ### **{best_label}**
            **Backstory:**  
            {matched_backstory}
            """)

    # Play audio
            if os.path.exists(matched_audio):
                audio_bytes = open(matched_audio, "rb").read()
                audio_box.audio(audio_bytes, format="audio/mp3")
            else:
                audio_box.write("Audio file not found!")

            st.session_state.last_label = best_label
        else:
            info_box.write("No object recognized yet.")
            audio_box.empty()
            st.session_state.last_label = None

# ---- back to OpenCV display ----
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