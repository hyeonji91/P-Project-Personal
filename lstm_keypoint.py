import cv2
import mediapipe as mp
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from st_gcn import STGCN
import random
from tqdm import tqdm
import time
import wandb


# 시드고정
def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_preprocessing(keypoints, num_person=1, num_channels=3):
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

    data = np.zeros((num_channels, num_frames, num_nodes, num_person))
    for t in range(num_frames):
        data[:, t, :, 0] = keypoints[t].T

    return torch.tensor(data, dtype=torch.float)


class SignLangDataSet(Dataset):
    def __init__(self, keypoints, labels, label_to_idx, max_len=30):
        """
        Args:
            keypoints (list): keypoint 리스트
            labels (list): 각 영상에 대한 레이블 리스트
        """
        self.keypoints = keypoints
        self.labels = labels
        self.label_to_idx = label_to_idx  # 문자열 -> 정수 변환 매핑
        self.max_len = max_len  # 최대 프레임 길이 설정

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        keypoints = self.keypoints[idx]
        label = self.labels[idx]
        #
        # keypoints = extract_keypoints(video_path) # (T, V, C) 형식 반환
        # keypoints = sample_keypoints(keypoints, self.max_len)  # (max_len, V, C)
        data = data_preprocessing(keypoints)  # (C, T, V, M) 형식으로 변환

        # 문자열 레이블 -> 정수 변환
        label_idx = self.label_to_idx[label]
        return data, torch.tensor(label_idx, dtype=torch.long)


### hyperparameter setting ###
batch_size = 400
epochs = 600
learning_rate = 0.01

### GPU Setting ###
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else 'cpu')
print(DEVICE)
set_env(42) # 시드고정

### setting wandb ###
wandb.init(project="sign-language-st-gcn",

           config = {
               "batch_size": batch_size,
               "epochs": epochs,
               "learning_rate" : learning_rate
           })


num_of_video = 3000


save_path = 'data/keypoint1~3000.npy'
keypoint_load = np.load(save_path)
print(keypoint_load.shape)


# label읽기
if DEVICE == torch.device("cuda"):
    df = pd.read_excel('/media/vom/HDD1/hj/p-project/KETI-2017-SL-Annotation-v2_1.xlsx')
else:
    df = pd.read_excel('F:/HyeonjiKim/Downloads/signLanguageDataset/KETI-2017-SL-Annotation-v2_1.xlsx')
df.sort_values(by = '번호', ascending=True, inplace=True)
label_list = df['한국어'].tolist()
label_list = np.array(label_list[:num_of_video])


### label - idx mapping정보 가져오기
import pickle
with open('data/label_to_idx.pickle', 'rb') as f:
    label_to_idx = pickle.load(f)
print(label_to_idx)


#데이터 생성
X_train, X_test, y_train, y_test = train_test_split(keypoint_load, label_list, test_size = 0.2, random_state=42, stratify=label_list)

train_dataset = SignLangDataSet(X_train, y_train, label_to_idx)
test_dataset = SignLangDataSet(X_test, y_test, label_to_idx)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

