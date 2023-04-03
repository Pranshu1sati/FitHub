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

def Pushups():
    up_pos = None
    down_pos = None
    pushup_pos = None
    display_pos = None
    push_up_counter = 0
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    pose = mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frames=cap.read()
        image =cv2.cvtColor(cv2.flip(frames,1),cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = pose.process(image)
        image.flags.writeable=True
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            Left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            Left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            Right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_arm_angle = int(calculate_angle(Left_shoulder, Left_elbow, Left_wrist))
            right_arm_angle = int(calculate_angle(Right_shoulder, Right_elbow, Right_wrist))

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))
            right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))
            
            # print(left_leg_angle)
            if left_arm_angle > 160 and left_leg_angle <167:
                up_pos = 'Up'
                display_pos = 'Up'

            if left_arm_angle < 110 and up_pos == 'Up':
                down_pos = 'Down'
                display_pos = 'Down'    

            if left_arm_angle > 160 and down_pos == 'Down':

                pushup_pos = "up"
                display_pos = "up"
                push_up_counter += 1

                up_pos = None
                down_pos = None
                pushup_pos = None  
            mp_draw.draw_landmarks(image, result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, str(push_up_counter), (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Feed',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
  
Pushups()