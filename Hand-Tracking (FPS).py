#This program uses the mediapipe and opencv library to track the user's hand motions and prints the fps on the top left corner.
#Author @Wei-cen Chen
#Version December 24, 2021

#Importing the necessary libraries to run the program
import cv2
import mediapipe as mp
import time

#Allows for recording
cap = cv2.VideoCapture(0)

#Declares variables to be used later
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

#Initializing variables to be used later
pTime = 0
cTime = 0

#Executing program to track hand and print fps
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    #If statement that prints a rectangle around hand if a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #Calculate the dimensions of the rectangles
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            #Tracks the landmark points of the hand and follows it
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #Performs calculations to determine frames per second
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #Prints fps on the top left corner of your screen
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
