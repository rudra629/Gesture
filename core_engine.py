import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

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
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Volume Dial Variables
last_angle = None
volume_sensitivity = 4  # Degrees you have to twist to trigger a volume change

frameR = 140  
base_smooth = 4       
aim_assist_smooth = 12  
speed_threshold = 20    

cap = cv2.VideoCapture(0)
cam_w, cam_h = 640, 480
cap.set(3, cam_w)
cap.set(4, cam_h)

print("Air Dashboard Engine Live.")
print("1. Steer with Open Palm.")
print("2. Pinch to Click/Drag.")
print("3. Closed Fist to Pause (Clutch).")
print("4. PEACE SIGN and Twist Wrist for Volume Dial.")

def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    for tip, joint in zip(tips, joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            return False
    return True

def is_peace_sign(hand_landmarks):
    # Index and Middle are UP (Y coordinate is lower than the joint)
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    # Ring and Pinky are DOWN (Y coordinate is higher than the joint)
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    return index_up and middle_up and ring_down and pinky_down

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (0, 255, 255), 2)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, c = img.shape
        wrist_x, wrist_y = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
        thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
        index_x, index_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
        palm_x, palm_y = int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)

        # -------------------------
        # 1. VOLUME DIAL LOGIC (Peace Sign)
        # -------------------------
        if is_peace_sign(hand_landmarks):
            cv2.putText(img, "VOLUME DIAL ACTIVE", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            
            # Calculate the angle of the hand
            current_angle = math.degrees(math.atan2(palm_y - wrist_y, palm_x - wrist_x))
            
            # Draw a visual line showing the "Dial"
            cv2.line(img, (wrist_x, wrist_y), (palm_x, palm_y), (255, 255, 0), 5)
            
            if last_angle is not None:
                angle_diff = current_angle - last_angle
                
                # Handle the math wrap-around at 180 / -180 degrees
                if angle_diff > 180: angle_diff -= 360
                if angle_diff < -180: angle_diff += 360
                
                # If twisted clockwise past the sensitivity threshold
                if angle_diff > volume_sensitivity:
                    pyautogui.press('volumeup')
                    print(">>> VOLUME UP")
                    last_angle = current_angle # Reset base angle
                
                # If twisted anti-clockwise
                elif angle_diff < -volume_sensitivity:
                    pyautogui.press('volumedown')
                    print("<<< VOLUME DOWN")
                    last_angle = current_angle # Reset base angle
            else:
                last_angle = current_angle
                
            # Make sure we drop a held click if we switched to volume mode
            if is_clicking:
                pyautogui.mouseUp()
                is_clicking = False

        # -------------------------
        # 2. CLUTCH LOGIC (Fist)
        # -------------------------
        elif is_fist(hand_landmarks):
            last_angle = None # Reset volume dial
            cv2.putText(img, "CLUTCH (PAUSED)", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if is_clicking:
                pyautogui.mouseUp()
                is_clicking = False
            
        # -------------------------
        # 3. MOVEMENT & CLICK LOGIC
        # -------------------------
        else:
            last_angle = None # Reset volume dial
            cv2.circle(img, (palm_x, palm_y), 15, (0, 255, 0), cv2.FILLED)
            
            screen_target_x = np.interp(palm_x, (frameR, cam_w - frameR), (0, screen_w))
            screen_target_y = np.interp(palm_y, (frameR, cam_h - frameR), (0, screen_h))

            hand_speed = math.hypot(screen_target_x - plocX, screen_target_y - plocY)
            
            if hand_speed < speed_threshold:
                active_smooth = aim_assist_smooth
                cv2.putText(img, "AIM ASSIST ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                active_smooth = base_smooth

            clocX = plocX + (screen_target_x - plocX) / active_smooth
            clocY = plocY + (screen_target_y - plocY) / active_smooth
            
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

            click_distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            if click_distance < 30:
                cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED) 
                if not is_clicking:
                    pyautogui.mouseDown()
                    is_clicking = True
            else:
                if is_clicking:
                    pyautogui.mouseUp()
                    is_clicking = False

    cv2.imshow("Air Dashboard OS Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if is_clicking: pyautogui.mouseUp()
        break

cap.release()
cv2.destroyAllWindows()