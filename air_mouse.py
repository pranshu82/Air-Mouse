import mediapipe as mp
import cv2
import time
import numpy as np
import math
# import autopy
import wx
from pynput.mouse import Button,Controller

mouse = Controller()
app = wx.App()

pTime = 0
wScr , hScr = wx.GetDisplaySize()
print(wScr,hScr)
#autopy.screen.size()
hCam = 480
wCam = 640
frameR = 100
smoothening = 7
p_loc_x ,p_loc_y = 0, 0
c_loc_x ,c_loc_y = 0, 0
toggle = True

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def finger_up(coords):
    fingers = []

    tips = [4,8,12,16,20]
    for i in tips:
        if i==4:
            if math.hypot(coords[i][0]-coords[5][0] ,coords[i][1]-coords[5][1]) > math.hypot(coords[i-2][0]-coords[5][0] ,coords[i-2][1]-coords[5][1]):
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if math.hypot(coords[i][0]-coords[0][0] ,coords[i][1]-coords[0][1]) > math.hypot(coords[i-2][0]-coords[0][0] ,coords[i-2][1]-coords[0][1]):
                fingers.append(1)
            else:
                fingers.append(0)

    return fingers

with mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.5,max_num_hands = 1) as hands:
    cap = cv2.VideoCapture(0) 
    pTime = 0
    while cap.isOpened():
        ret,frame = cap.read()
        image = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                coords = []
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)
                for i in range(21):
                    x,y = int((hand.landmark[i].x)*640),int((hand.landmark[i].y)*480)
                    coords.append([x,y])

                x1,y1 = coords[8][0],coords[8][1]
                x2,y2 = coords[12][0],coords[12][1]
                cx,cy = (x1+x2)//2,(y1+y2)//2

                cv2.rectangle(image,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

                finger = finger_up(coords)
                if finger[1] == 1 and finger[2] == 0:
                    # mouse.release(Button.left)

                    x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
                    y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
                    c_loc_x = p_loc_x + (x3-p_loc_x)/smoothening
                    c_loc_y = p_loc_y + (y3-p_loc_y)/smoothening


                    # autopy.mouse.move(wScr - c_loc_x,c_loc_y)
                    mouse.position = (c_loc_x,c_loc_y)
                    
                    cv2.circle(image,(x1,y1),15,(255,0,255),cv2.FILLED)
                    p_loc_x,p_loc_y = c_loc_x,c_loc_y

                if finger[1] == 1 and finger[2] == 1:
                    cv2.circle(image,(x1,y1),10,(255,0,255),cv2.FILLED)
                    cv2.circle(image,(cx,cy),10,(0,0,255),cv2.FILLED)
                    cv2.line(image,(x1,y1),(x2,y2),(255,0,255),3)
                    cv2.circle(image,(x2,y2),10,(255,0,255),cv2.FILLED)

                    length = math.hypot(x1-x2,y1-y2)
                    # mouse.release(Button.left)
                    if length < 20:
                        cv2.circle(image,(cx,cy),10,(0,255,0),cv2.FILLED)
                        # autopy.mouse.click()
                        if toggle == True:
                            mouse.click(Button.left,1)
                            toggle = False
                    else:
                        toggle = True
                        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(image,str(int(fps)),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)

        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()