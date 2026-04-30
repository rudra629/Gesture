import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from collections import deque

# PyAutoGUI Setup
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  
screen_w, screen_h = pyautogui.size()

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# State Variables
is_clicking = False
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

frameR = 100  

cap = cv2.VideoCapture(0)
cam_w, cam_h = 640, 480
cap.set(3, cam_w)
cap.set(4, cam_h)

print("Air Dashboard Engine Live.")
print("1. Move OPEN PALM to steer cursor.")
print("2. PINCH (Index + Thumb) to Click or Hold-to-Drag.")
print("3. Make a CLOSED FIST to pause tracking (The Clutch).")

# Helper function to detect a closed fist
def is_fist(hand_landmarks):
    # Tip landmarks: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
    # Joint landmarks: 6 (Index), 10 (Middle), 14 (Ring), 18 (Pinky)
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    
    # In OpenCV, Y coordinates increase as you go down the screen.
    # If the tips are lower than the joints, the fingers are folded (fist).
    for tip, joint in zip(tips, joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            return False  # A finger is extended
    return True

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, c = img.shape
        
        thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
        index_x, index_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
        palm_x, palm_y = int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)

        # -------------------------
        # 1. CHECK FOR FIST (The Clutch)
        # -------------------------
        if is_fist(hand_landmarks):
            # If fist is detected, release click if held, and freeze movement
            cv2.putText(img, "SYSTEM PAUSED (CLUTCH)", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if is_clicking:
                pyautogui.mouseUp()
                is_clicking = False
            
        else:
            # -------------------------
            # 2. MOVEMENT LOGIC (Track Palm)
            # -------------------------
            cv2.circle(img, (palm_x, palm_y), 15, (0, 255, 0), cv2.FILLED)
            
            screen_target_x = np.interp(palm_x, (frameR, cam_w - frameR), (0, screen_w))
            screen_target_y = np.interp(palm_y, (frameR, cam_h - frameR), (0, screen_h))

            clocX = plocX + (screen_target_x - plocX) / smoothening
            clocY = plocY + (screen_target_y - plocY) / smoothening
            
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

            # -------------------------
            # 3. CLICK / DRAG LOGIC (Pinch)
            # -------------------------
            click_distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            if click_distance < 30:
                cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED) # Visual feedback for click
                if not is_clicking:
                    # NOTE: Change 'left' to 'right' below if you strictly want right-click dragging
                    pyautogui.mouseDown(button='left')
                    is_clicking = True
                    print(">>> MOUSE DOWN (Drag Initiated)")
            else:
                if is_clicking:
                    pyautogui.mouseUp(button='left')
                    is_clicking = False
                    print("<<< MOUSE UP (Drag Released / Click Complete)")

    cv2.imshow("Air Dashboard OS Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Safety net: Ensure mouse is released on exit
        if is_clicking:
            pyautogui.mouseUp()
        break

cap.release()
cv2.destroyAllWindows()