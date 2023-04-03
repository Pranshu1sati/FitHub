import time
import cv2
from flask import Flask, render_template, Response
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


app =Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
def gen():
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.5)
    counter = 0 
    stage = None
    flag = False
    half = 10
    text=""
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
# with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img,1)
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        # print(result.pose_landmarks)
        if result.pose_landmarks:
            flag = True if counter%2 == 0 else False
            landmarks = result.pose_landmarks.landmark
            if flag:
                shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]
            else:
                shoulder = [landmarks[my_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[my_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[my_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[my_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angle = calculate_angle(shoulder,elbow,wrist)
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
            cv2.putText(img, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 2, cv2.LINE_AA
                                )
            mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS,
                                mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # checking video frame rate
        # current_time = time.time()
        # fps = 1 / (current_time - previous_time)
        # previous_time = current_time

        # Writing FrameRate on video
        if(counter==0):
            cv2.putText(img, str(int(counter)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        elif(counter != 0 and  counter%20)==0:
            cv2.putText(img, str("Good Job"), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        else:
            cv2.putText(img, str(int(counter)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break
# def genYoga()

def generate_frames():
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    """Video streaming generator function."""
    label="Unknown Pose"
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        img = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        # print(result.pose_landmarks)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            Left_shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Left_elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
            Left_wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]
            Left_elbowAngle = calculate_angle(Left_shoulder,Left_elbow,Left_wrist)
            Right_shoulder = [landmarks[my_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Right_elbow = [landmarks[my_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Right_wrist = [landmarks[my_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[my_pose.PoseLandmark.RIGHT_WRIST.value].y]
            Right_elbowAngle = calculate_angle(Right_shoulder,Right_elbow,Right_wrist)
            Left_hip = [landmarks[my_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[my_pose.PoseLandmark.LEFT_HIP.value].y]
            Right_hip = [landmarks[my_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[my_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_shoulder_angle = calculate_angle(Left_elbow,
                                         Left_shoulder,
                                         Left_hip)

    # # Get the angle between the right hip, shoulder and elbow points. 
            right_shoulder_angle = calculate_angle(Right_hip,
                                          Right_shoulder,
                                          Right_elbow)
            if Left_elbowAngle > 165 and Left_elbowAngle < 195 and Right_elbowAngle > 165 and Right_elbowAngle < 195:
      
                if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
                        label = "T pose"

                else:
                    label='straiten your arms & shoulders'
            else:
                label = 'extend your arms in opp //n direction'
            cv2.putText(img, str(Left_elbowAngle), 
                           tuple(np.multiply(Left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(img, str(Right_elbowAngle), 
                           tuple(np.multiply(Right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 2, cv2.LINE_AA
                                )
            mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)

        # checking video frame rate
        # current_time = time.time()
        # fps = 1 / (current_time - previous_time)
        # previous_time = current_time

        # Writing FrameRate on video
        cv2.putText(img, str( label), (60, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


def genTree():
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    """Video streaming generator function."""
    label="Unknown Pose"
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img,1)
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        # print(result.pose_landmarks)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # Left_shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # Left_elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
            # Left_wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_knee_angle = calculate_angle([landmarks[my_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[my_pose.PoseLandmark.LEFT_HIP.value].y],
                                    [ landmarks[my_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[my_pose.PoseLandmark.LEFT_KNEE.value].y],
                                     [landmarks[my_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[my_pose.PoseLandmark.LEFT_ANKLE.value].y])
            
                
            right_knee_angle = calculate_angle([landmarks[my_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[my_pose.PoseLandmark.RIGHT_HIP.value].y],
                                      [landmarks[my_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[my_pose.PoseLandmark.RIGHT_KNEE.value].y],
                                      [landmarks[my_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[my_pose.PoseLandmark.RIGHT_ANKLE.value].y])
            


            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
                if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
                     label = 'Tree Pose'
                else:
                    label = "Unkown Pose"
            if right_knee_angle > 165 and right_knee_angle < 195 or left_knee_angle > 165 and left_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
                if right_knee_angle > 315 and right_knee_angle < 335 or left_knee_angle > 25 and left_knee_angle < 45:
                        label = "Tree Pose"
                else:
                    label = "Unknown Pose"
            else:
                label = 'Unknown Pose'
            
            # cv2.putText(img, str(Left_elbowAngle), 
            #                tuple(np.multiply(Left_elbow, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(img, str(Right_elbowAngle), 
            #                tuple(np.multiply(Right_elbow, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 2, cv2.LINE_AA
            #                     )
            mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)

        # checking video frame rate
        # current_time = time.time()
        # fps = 1 / (current_time - previous_time)
        # previous_time = current_time

        # Writing FrameRate on video
        cv2.putText(img, str( label), (60, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break
#  if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

#         # Check if the other leg is bended at the required angle.
#         if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

#             # Specify the label of the pose that is tree pose.
#             label = 'Tree Pose'
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
        frames = cv2.flip(frames,1)
        # image.flags.writeable = False
        result = pose.process(image)
        # image.flags.writeable=True
        
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
                    # print(push_up_counter)
                up_pos = None
                down_pos = None
                pushup_pos = None  
            mp_draw.draw_landmarks(frames, result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(frames, str(push_up_counter), (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            frame = cv2.imencode('.jpg', frames)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break


def Tricep():
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
        frames = cv2.flip(frames,1)
        # image.flags.writeable = False
        result = pose.process(image)
        # image.flags.writeable=True
        
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

            # left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            # right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            # left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            # right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            # left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            # right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            # left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))
            # right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))
            # print(left_arm_angle,right_arm_angle )
            # print(left_arm_angle,right_arm_angle)
            if left_arm_angle > 140 and right_arm_angle>140:
                up_pos = 'Up'
                display_pos = 'Up'

            if left_arm_angle < 40 and right_arm_angle<40 and up_pos == 'Up':
                down_pos = 'Down'
                display_pos = 'Down'    

            if left_arm_angle > 140 and right_arm_angle>140 and down_pos == 'Down':

                pushup_pos = "up"
                display_pos = "up"
                push_up_counter += 1
                    # print(push_up_counter)
                up_pos = None
                down_pos = None
                pushup_pos = None  
            mp_draw.draw_landmarks(frames, result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(frames, str(push_up_counter), (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            frame = cv2.imencode('.jpg', frames)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break

@app.route('/curl')
 
def curl():
    return render_template('curl.html')

@app.route('/yoga')

def yoga():
    return render_template('yoga.html')


@app.route('/tree')
def tree(): 
    return render_template('tree.html')

@app.route('/about')
def about(): 
    return render_template('about.html') 
@app.route('/pushups')
def pushups():
        return render_template('pushups.html')
@app.route('/tricep')
def tricep():
    return render_template('tricep.html')

#-----------------Video-Feed---------------------------------
@app.route('/curl_video')

def curl_video():
    return Response(gen(),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/yoga_video')

def yoga_video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tree_video') 
def tree_video():
    return Response(genTree(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/pushups_video')
def pushups_video():
    return Response(Pushups(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/tricep_video')
def tricep_video():
    return Response(Tricep(),mimetype='multipart/x-mixed-replace; boundary=frame')
#--------------------------------------------------------------
app.run(debug=True)