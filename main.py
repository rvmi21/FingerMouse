import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui as pg
import math

HCam = 480
WCam = 640
HScrn = 1080
WScrn = 1920

#camera object
cap = cv2.VideoCapture(0)

#hand recognizer
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
#set timer
pTime = 0
cTime = 0



Index_tip_y = -1
Index_tip_x = -1
Index_dip_y = -1


middle_dinge_x = -1
middle_dinge_y = -1
middle_y = -1

thumb_x = -1
thumb_y = -1
thux = -1

fingr = [0, 0, 0]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #draw landmarks on hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                if id ==8:
                    Index_tip_y = cy
                    Index_tip_x = cx
                if id ==6:
                    Index_dip_y = cy
                    Index_dip_x = cx
                if id == 4:
                    thumb_x = cx
                    thumb_y = cy
                if id == 12:
                    middle_dinge_x = cx
                    middle_dinge_y = cy
                if id == 10:
                    middle_y = cy
                if id == 2:
                    thux = cy ;

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if(Index_dip_y>Index_tip_y):
        fingr[1] = 1
    else:
        fingr[1] = 0

    if (middle_y > middle_dinge_y):
        fingr[2] = 1
    else:
        fingr[2] = 0

    if (thux < thumb_x):
        fingr[0] = 1
    else:
        fingr[0] = 0

    if(fingr[1]==1) and (fingr[2]==0):

        cv2.rectangle(img, (100,100),(WCam-100,HCam-100),(200,100,100),2)

        xMouse = np.interp(Index_tip_x ,(100,WCam-100), (0,WScrn))
        yMouse = np.interp(Index_tip_y ,(100,HCam-100), (0,HScrn))

        pg.moveTo(xMouse, yMouse)

    if (fingr[1] == 1) and (fingr[2] == 1):
        distance = pow(Index_tip_x - middle_dinge_x,2) + pow(Index_tip_y - middle_dinge_y,2)
        distance = pow(distance,1/2)
        if distance < 25 :
            pg.click()
    #print(fingr)

        # fps calculator
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)



    cv2.imshow("Image", img)
    cv2.waitKey(1)
