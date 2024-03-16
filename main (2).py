import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
video = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# cos θ = (a * b) / |a||b| --> Formula to Find θ between 2 vectors
def angle(point1, point2, point3):
    vector1 = np.array([point2.x, point2.y, point2.z]) - np.array([point1.x, point1.y, point1.z])
    vector2 = np.array([point3.x, point3.y, point3.z]) - np.array([point1.x, point1.y, point1.z])
    angle_deg = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    return angle_deg


def three_finger_solute(fingerAngles):
    return (fingerAngles[0] < 20) and (fingerAngles[1] < 20) and (fingerAngles[2] < 20) and \
        (fingerAngles[3] > 10) and (fingerAngles[4] > 20)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while video.isOpened():
        success, frame = video.read()
        frame, frame2 = cv2.resize(frame, (550, 380)), cv2.resize(frame, (550, 380))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultPose, resultHands = pose.process(rgb), hands.process(rgb)
        if resultPose.pose_landmarks:
            topHead = resultPose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].y
            mp_drawing.draw_landmarks(frame, resultPose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        if resultHands.multi_hand_landmarks and resultPose.pose_landmarks:
            for hand_landmarks in resultHands.multi_hand_landmarks:
                if hand_landmarks.landmark[0].y < topHead:
                    finger_angles = [
                        angle(hand_landmarks.landmark[8], hand_landmarks.landmark[6], hand_landmarks.landmark[5]),
                        angle(hand_landmarks.landmark[12], hand_landmarks.landmark[10], hand_landmarks.landmark[9]),
                        angle(hand_landmarks.landmark[16], hand_landmarks.landmark[14], hand_landmarks.landmark[13]),
                        angle(hand_landmarks.landmark[4], hand_landmarks.landmark[3], hand_landmarks.landmark[1]),
                        angle(hand_landmarks.landmark[20], hand_landmarks.landmark[18], hand_landmarks.landmark[17])
                    ]
                    if three_finger_solute(finger_angles):
                        frame2 = cv2.putText(frame2, 'I volunteer as tribute', (150, 50),
                                             cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)
                    frame = cv2.putText(frame, 'Hand is Raised', (150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2,
                                        cv2.LINE_AA)

            for hand_landmarks in resultHands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        cv2.imshow('Raise Hand Detecting...', frame)
        cv2.imshow('Volunteer Tribute Detecting...', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
