import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mphands = mp.solutions.hands
hands= mphands.Hands()

mpdraw = mp.solutions.drawing_utils

ctime = 0
ptime = 0

while True:
    success , img = cap.read()
    imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
   # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id , lm in enumerate(handlms.landmark):
                #print(id , lm)
                h , w , c = img.shape
                cx , cy = int(lm.x * w ) , int(lm.y * h)
                if id == 4 or id ==0 or id == 8 or id ==12 or id ==16 or id ==20:
                    cv.circle(img , (cx , cy) , 10, (0 , 255 , 0) , cv.FILLED)


            mpdraw.draw_landmarks(img , handlms , mphands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv.putText(img , str(int(fps)) , (10 , 70) , cv.FONT_HERSHEY_PLAIN , 3 ,
               (255 , 0 , 255) , 3 , 3)
    cv.imshow("image", img)
    cv.waitKey(1)
