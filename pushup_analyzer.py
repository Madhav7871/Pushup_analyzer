import cv2
import mediapipe as mp
import numpy as np

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
# Higher confidence prevents "ghost" movements from counting
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Logic variables - Strict State Machine
counter = 0 
stage = None # This remains None until a clear 'up' or 'down' is detected
feedback = "GET READY"
ui_color = (0, 255, 0) 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Header Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1) 
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Using Left side keypoints
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            back_angle = calculate_angle(shoulder, hip, ankle)
            
            # Progress calculation
            per = np.interp(elbow_angle, (70, 160), (100, 0))

            # --- THE PERFECT COUNTING LOGIC ---
            # 1. Check for back alignment first
            if back_angle < 150:
                feedback = "KEEP BACK STRAIGHT"
                ui_color = (0, 0, 255) # Red
            else:
                ui_color = (0, 255, 0) # Green
                # 2. Must reach deep 'down' to trigger stage
                if elbow_angle <= 80:
                    if stage != "down":
                        feedback = "GO UP!"
                    stage = "down"
                
                # 3. Must return to full 'up' extension to count
                if elbow_angle >= 160 and stage == "down":
                    stage = "up"
                    counter += 1
                    feedback = "PERFECT!"
                elif elbow_angle >= 160:
                    stage = "up"
                    feedback = "LOWER YOUR CHEST"

            # --- EXACT IMAGE 2 STYLING ---
            # Drawing the skeleton with WHITE lines and RED dots (matching your image 2)
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=6), # Red Dots
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2) # White Lines
            )

            # --- UI ELEMENTS ---
            # Circular Counter (Top Left)
            cv2.circle(frame, (80, 50), 40, ui_color, -1)
            cv2.putText(frame, str(counter), (60, 65), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, "REPS", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Feedback Text (Center)
            cv2.putText(frame, feedback, (w//2 - 120, 60), cv2.FONT_HERSHEY_DUPLEX, 1, ui_color, 2)
            
            # Angle Boxes (Bottom)
            cv2.rectangle(frame, (20, h-70), (200, h-20), (20, 20, 20), -1)
            cv2.putText(frame, f"ELBOW: {int(elbow_angle)}", (30, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (w-200, h-70), (w-20, h-20), (20, 20, 20), -1)
            cv2.putText(frame, f"BACK: {int(back_angle)}", (w-180, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
        except Exception as e:
            pass
    else:
        feedback = "STEP INTO FRAME"

    cv2.imshow('Modern AI Trainer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
# hello world!
cap.release()
cv2.destroyAllWindows()