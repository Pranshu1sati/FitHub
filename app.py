import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
import numpy as np
# from flask import Flask , render_template, Response
# from camera import Video
# def calculateAngle2(landmark1, landmark2, landmark3):
#     '''
#     This function calculates angle between three different landmarks.
#     Args:
#         landmark1: The first landmark containing the x,y and z coordinates.
#         landmark2: The second landmark containing the x,y and z coordinates.
#         landmark3: The third landmark containing the x,y and z coordinates.
#     Returns:
#         angle: The calculated angle between the three landmarks.

#     '''

#     # Get the required landmarks coordinates.
#     x1, y1, _ = landmark1
#     x2, y2, _ = landmark2
#     x3, y3, _ = landmark3

#     # Calculate the angle between the three points
#     angle = np.degrees(np.atan2(y3 - y2, x3 - x2) - np.atan2(y1 - y2, x1 - x2))
    
#     # Check if the angle is less than zero.
#     if angle < 0:

#         # Add 360 to the found angle.
#         angle += 360
    
#     # Return the calculated angle.
#     return angle


# # def classifyPose(landmarks, output_image, display=False):
#      # Initializing mediapipe pose class.

#     '''
#     This function classifies yoga poses depending upon the angles of various body joints.
#     Args:
#         landmarks: A list of detected landmarks of the person whose pose needs to be classified.
#         output_image: A image of the person with the detected pose landmarks drawn.
#         display: A boolean value that is if set to true the function displays the resultant image with the pose label 
#         written on it and returns nothing.
#     Returns:
#         output_image: The image with the detected pose landmarks drawn and pose label written.
#         label: The classified pose label of the person in the output_image.

#     '''
    
#     # Initialize the label of the pose. It is not known at this stage.
#     label = 'Unknown Pose'

#     # Specify the color (Red) with which the label will be written on the image.
#     color = (0, 0, 255)
    
#     # Calculate the required angles.
#     #----------------------------------------------------------------------------------------------------------------
    
#     # Get the angle between the left shoulder, elbow and wrist points. 
#     mp_pose = mp.solutions.pose

#         # Setting up the Pose function.
#     pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

#         # Initializing mediapipe drawing class, useful for annotation.
#     mp_drawing = mp.solutions.drawing_utils 
#     left_elbow_angle = calculateAngle2(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
#     # Get the angle between the right shoulder, elbow and wrist points. 
#     right_elbow_angle = calculateAngle2(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
#     # Get the angle between the left elbow, shoulder and hip points. 
#     left_shoulder_angle = calculateAngle2(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

#     # Get the angle between the right hip, shoulder and elbow points. 
#     right_shoulder_angle = calculateAngle2(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

#     # Get the angle between the left hip, knee and ankle points. 
#     left_knee_angle = calculateAngle2(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
#                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

#     # Get the angle between the right hip, knee and ankle points 
#     right_knee_angle = calculateAngle2(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
#                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
#     #----------------------------------------------------------------------------------------------------------------
    
#     # Check if it is the warrior II pose or the T pose.
#     # As for both of them, both arms should be straight and shoulders should be at the specific angle.
#     #----------------------------------------------------------------------------------------------------------------
    
#     # Check if the both arms are straight.
#     if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

#         # Check if shoulders are at the required angle.
#         if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

#     # Check if it is the warrior II pose.
#     #----------------------------------------------------------------------------------------------------------------

#             # Check if one leg is straight.
#             if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

#                 # Check if the other leg is bended at the required angle.
#                 if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

#                     # Specify the label of the pose that is Warrior II pose.
#                     label = 'Warrior II Pose' 
                        
#     #----------------------------------------------------------------------------------------------------------------
    
#     # Check if it is the T pose.
#     #----------------------------------------------------------------------------------------------------------------
    
#             # Check if both legs are straight
#             if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

#                 # Specify the label of the pose that is tree pose.
#                 label = 'T Pose'

#     #----------------------------------------------------------------------------------------------------------------
    
#     # Check if it is the tree pose.
#     #----------------------------------------------------------------------------------------------------------------
    
#     # Check if one leg is straight
#     if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

#         # Check if the other leg is bended at the required angle.
#         if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

#             # Specify the label of the pose that is tree pose.
#             label = 'Tree Pose'
                
#     #----------------------------------------------------------------------------------------------------------------
    
#     # Check if the pose is classified successfully
#     if label != 'Unknown Pose':
        
#         # Update the color (to green) with which the label will be written on the image.
#         color = (0, 255, 0)  
    
#     # Write the label on the output image. 
#     cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
#     # Check if the resultant image is specified to be displayed.
#     # if display:
    
#     #     # Display the resultant image.
#     #     plt.figure(figsize=[10,10])
#     #     plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
#     # else:
        
#     #     # Return the output image and the classified label.
#     return label


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
#--------------------------------------------------------------
app.run(debug=True)