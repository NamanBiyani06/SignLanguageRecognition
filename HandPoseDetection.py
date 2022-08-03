import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

#Draws line segments
mp_drawing = mp.solutions.drawing_utils
#mp hands model
mp_hands = mp.solutions.hands
#styles for drawing hands
mp_drawing_styles = mp.solutions.drawing_styles

#getting webcam feed #0
cap = cv2.VideoCapture(0)

cv2.startWindowThread()

#declaring following as "hands"
#detection threshold for intitial detection to be successful
#tracking threshold for tracking after initial detection
with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    #reading through each frame
    while cap.isOpened():
        ret,frame = cap.read()

        #detections
        #recolours from BGR to RGB
        #passes in frame, outputs image and RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_height, image_width, c = image.shape

        #image is flipped on the horizontal
        image = cv2.flip(image, 1)

        image.flags.writeable = False
        #detections
        results = hands.process(image)
        image.flags.writeable = True
        #RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #checking if we have results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
            #printing each of the 20 landmarks    
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                print(f'HAND NUMBER: {hand_no+1}')
                print('-----------------------')
                
                for i in range(20):    
                  print(f'{mp_hands.HandLandmark(i).name}:') 
                  print(f'x: {hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width}')
                  print(f'y: {hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height}')

                
        cv2.imshow('Hand Tracking', image)

        #close window if q is pressed
        if cv2.waitKey(10) & 0xff == ord('q'):
            for i in range(1,10):
                cv2.destroyAllWindows()
            break
        
cv2.destroyAllWindows()
cap.release()

mp_hands.HAND_CONNECTIONS
