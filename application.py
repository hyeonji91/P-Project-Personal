import cv2
import mediapipe as mp
import numpy as np
import torch
from st_gcn import STGCN



def extract_keypoints(result):

    # 몸, 손 키포인트 추출
    if results.pose_landmarks:
        pose = np.array([[kp.x, kp.y, kp.z] for kp in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))  # 33개 keypoints
    if results.left_hand_landmarks:
        left_hand = np.array([[kp.x, kp.y, kp.z] for kp in results.left_hand_landmarks.landmark])
    else:
        left_hand = np.zeros((21, 3))
    if results.right_hand_landmarks:
        right_hand = np.array([[kp.x, kp.y, kp.z] for kp in results.right_hand_landmarks.landmark])
    else:
        right_hand = np.zeros((21, 3))

    frame_keypoints = np.concatenate([pose, left_hand, right_hand])

    return frame_keypoints

def data_preprocessing(keypoints, num_person = 1, num_channels = 3):
    """
    ST-GCN 입력 형식으로 키포인트 데이터를 변환합니다.
    Args:
        keypoints: (num_frames, num_nodes, 3) 형식의 키포인트 데이터
        num_person: 프레임당 사람의 수 (기본값: 1)
        num_channels: 좌표 차원 수 (기본값: 3 - x, y, z)
    Returns:
        Tensor: (C, T, V, M) 형식의 데이터
    st gcn 입력
        Tensor: (N, C, T, V, M) 형식의 데이터
        N : batch_size
        C : keypoint의 차원
        T : fps
        V : 한 프레임당 keypoint 개수
        M : 사람 수
    """
    num_frames, num_nodes, _ = keypoints.shape

    data = np.zeros((1, num_channels, num_frames, num_nodes, num_person))
    for t in range(num_frames):
        data[0, :, t, :, 0] = keypoints[t].T

    return torch.tensor(data, dtype=torch.float)

# 비디오
cap = cv2.VideoCapture(0)

# holistic설정
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

### 모델 가져오기 ###
graph_args = {"layout": "mediapipe", "strategy": "spatial"}
loaded_model = STGCN(in_channels=3, num_class=420, graph_args=graph_args, edge_importance_weighting=True)
loaded_model.load_state_dict(torch.load("model/best_model_4.pth", map_location=torch.device('cpu')))

keypoint_sequence = []

# 이미지 입력 캡처 및 처리
# media pipe 는 RGB
while cap.isOpened():
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(imageRGB)

    # print("왼손 랜드마크: ", results.left_hand_landmarks)
    # print("오른손 랜드마크: ", results.right_hand_landmarks)
    # print("얼굴 랜드마크: ", results.face_landmarks)
    # print("pose 랜드마크: ", results.pose_landmarks)

    keypoints = extract_keypoints(results)
    keypoint_sequence.append(keypoints)
    sequence = keypoint_sequence[-30:]  # 마지막 30 frame으로 prediction 한다

    if len(sequence) == 30:  # 30 프레임
        print(np.shape(sequence))

        output = loaded_model(data_preprocessing(np.array(sequence)))
        prediction = torch.argmax(output, dim=1)
        print('prediction ', prediction)



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

    cv2.imshow('output', annotated_image)
    cv2.waitKey(1)



cap.release()
holistic.close()

