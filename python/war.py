def Warrior():
         mpDraw = mp.solutions.drawing_utils
            # creating our model to detected our pose
         mp_pose = mp.solutions.pose
         pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
         label = ""
         cap = cv2.VideoCapture(0)
         while cap.isOpened():
             success, img = cap.read()
             img = cv2.flip(img,1)
             imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
             result = pose.process(imgRGB)
             if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark            
                left_shoulder_angle = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow_angle = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow_angle = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_shoulder_angle = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_knee_angle = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee_angle = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

            # Check if shoulders are at the required angle.
                  if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

        # Check if it is the warrior II pose.
        #----------------------------------------------------------------------------------------------------------------

                # Check if one leg is straight.
                      if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                    # Check if the other leg is bended at the required angle.
                         if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                        # Specify the label of the pose that is Warrior II pose.
                            label = 'Warrior II Pose' 
                mpDraw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
             cv2.putText(img, str( label), (60, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
             frame = cv2.imencode('.jpg', img)[1].tobytes()
             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
             key = cv2.waitKey(20)
             if key == 27:
                 break