import cv2
import mediapipe as mp

# 가져오기 및 초기화
cap = cv2.VideoCapture("/media/vom/HDD1/hj/p-project/0001~3000(video)/KETI_SL_0000000001.avi")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


# 이미지 입력 캡처 및 처리
# media pipe 는 RGB
while cap.isOpened():
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(imageRGB)

    print("왼손 랜드마크: ", results.left_hand_landmarks)
    print("오른손 랜드마크: ", results.right_hand_landmarks)
    print("얼굴 랜드마크: ", results.face_landmarks)
    print("pose 랜드마크: ", results.pose_landmarks)

    # 점 그리기
    annotated_image = image.copy()
    mp_draw.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_draw.draw_landmarks(
    #     annotated_image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style()
    #     )
    mp_draw.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
         landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style()
         )
    # 양손 분리
    # if results.multi_hand_landmarks:
    #     for handLms in results.multi_hand_landmarks: # 한 번에 한 손씩 작업
    #         for id, lm in enumerate(handLms.landmark):
    #             h,w,c = image.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             print("lmx : ", lm.x, ' lmy : ', lm.y)

    #             # 손 랜드마크 그리기
    #             if id == 20:
    #                cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
    #         mp_draw.draw_landmarks(image, handLms, mp_holistic.HAND_CONNECTIONS)
    cv2.imshow('output', annotated_image)
    cv2.waitKey(1)
cap.release()