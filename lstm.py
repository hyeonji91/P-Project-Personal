import torch.nn as nn
import torch
import torch.nn.functional as F


class lstm(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, hidden_size = 64):
        """
        :param input_dim: 각 시점에서 입력의 특징(feature) 수 (예: 키포인트 좌표, 관절 정보 등)
        :param num_classes: class 개수
        :param num_layers: ?
        :param hidden_size: ?
        """
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        LSTM input:
            batch_size: 한 번에 처리할 데이터의 수 (예: 32, 64 등, DataLoader에서 설정 가능)
            seq_len: 시계열 데이터의 길이 (예: 30 프레임)
            input_dim: 각 시점에서 입력의 특징(feature) 수 (예: 키포인트 좌표, 관절 정보 등)
        """
        # 최초의 hidden state와 cell state를 초기화시켜주어야 합니다.
        # 배치 사이즈는 가변적이므로 클래스 내에선 표현하지 않습니다.

        # LSTM 순전파
        out, _ = self.lstm(x) # output : (batch, seq_len, hidden_size) tensors. (hn, cn)은 필요 없으므로 받지 않고 _로 처리합니다.

        # 마지막 time step(sequence length)의 hidden state를 사용해 Class들의 logit을 반환합니다(hidden_size -> num_classes).
        out = self.fc(out[:, -1, :]) # (batch, hidden_size)

        # Fully connected layers with dropout
        x = self.dropout1(out)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # Output layer
        x = F.softmax(x, dim=1)  # Softmax for probabilities

        return x