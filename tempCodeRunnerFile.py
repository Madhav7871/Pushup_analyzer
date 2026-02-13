import cv2
import mediapipe as mp
import numpy as np

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Logic variables
counter = 0 
stage = "up" # Initial state
feedback = "START"
ui_color = (255, 191, 0) # Start with Deep Blue

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip the image for a 'mirror' effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Process Frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Create a UI Overlay Layer (Transparent Effect)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (40, 40, 40), -1) # Top Header
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Extract Keypoints (Using Left side for profile view)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculations
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            back_angle = calculate_angle(shoulder, hip, ankle)
            
            # Smooth Depth Progress (Circular)
            per = np.interp(elbow_angle, (70, 160), (100, 0))

            # --- SMOOTH STATE MACHINE ---
            if back_angle < 155:
                feedback = "STRAIGHTEN BACK"
                ui_color = (0, 0, 255) # Red for error
            elif per > 90:
                if stage == "up":
                    feedback = "PUSH UP!"
                    ui_color = (0, 255, 0) # Green for deep pushup
                    stage = "down"
            elif per < 10:
                if stage == "down":
                    counter += 1
                    stage = "up"
                    feedback = "GOOD REP!"
                    ui_color = (255, 191, 0) # Reset to Blue
            else:
                feedback = "KEEP GOING"

            # --- DESIGNED FRAME ELEMENTS ---
            
            # 1. Counter Circle (Top Left)
            cv2.circle(frame, (80, 50), 40, ui_color, -1)
            cv2.putText(frame, str(counter), (65, 65), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, "REPS", (62, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 2. Status & Feedback (Top Center)
            cv2.putText(frame, feedback, (w//2 - 100, 60), cv2.FONT_HERSHEY_DUPLEX, 1, ui_color, 2)
            
            # 3. Angle Meters (Bottom Corners)
            # Left: Elbow
            cv2.rectangle(frame, (20, h-80), (220, h-20), (30,30,30), -1)
            cv2.putText(frame, f"ELBOW: {int(elbow_angle)}*", (30, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Right: Back
            cv2.rectangle(frame, (w-220, h-80), (w-20, h-20), (30,30,30), -1)
            cv2.putText(frame, f"BACK: {int(back_angle)}*", (w-200, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 4. Custom Skeleton Design (Dots only for a cleaner look)
            for landmark in results.pose_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, ui_color, -1)
                
        except Exception as e:
            pass
    else:
        cv2.putText(frame, "POSING...", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Modern AI Trainer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()