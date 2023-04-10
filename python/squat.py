import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle>180.0:
        angle = 360-angle
    return angle

def Squat():
    up_pos = None
    down_pos = None
    squat_pos = None
    display_pos = None
    squat_counter = 0
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    pose = mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frames=cap.read()
        image =cv2.cvtColor(cv2.flip(frames,1),cv2.COLOR_BGR2RGB)
        frames = cv2.flip(frames,1)
        
        result = pose.process(image)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            
            Left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            Right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            Left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            Right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            Left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            Right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_leg_angle = int(calculate_angle(Left_hip, Left_knee, Left_ankle))
            right_leg_angle = int(calculate_angle(Right_hip, Right_knee, Right_ankle))

            if left_leg_angle < 100 and right_leg_angle < 100:
                down_pos = 'Down'
                display_pos = 'Down'

            if left_leg_angle > 160 and right_leg_angle > 160 and down_pos == 'Down':
                squat_pos = 'Up'
                display_pos = 'Up'
                squat_counter += 1
                down_pos = None
                squat_pos = None

            mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, str(squat_counter), (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('Feed',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

Squat()
