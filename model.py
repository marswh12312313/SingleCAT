import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        attention = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, V)
        return out


class SingleCellClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SingleCellClassifier, self).__init__()
        self.num_features = num_features  # 将 num_features 设置为实例属性

        # 卷积层（增加了神经元数量）
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 1024, 3, padding=1)
        self.bn6 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Conv1d(1024, 2048, 3, padding=1)
        self.bn7 = nn.BatchNorm1d(2048)
        self.conv8 = nn.Conv1d(2048, 4096, 3, padding=1)
        self.bn8 = nn.BatchNorm1d(4096)
        self.pool = nn.MaxPool1d(2)

        # 自注意力层
        self.attention1 = SelfAttention(4096)

        # 计算全连接层的输入大小
        self.fc_input_size = 4096 * (self.num_features // 256)

        # 全连接层定义
        self.fc1 = nn.Linear(self.fc_input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        # 注意力层和其他全连接层定义...
        self.attention2 = SelfAttention(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 应用卷积层和最大池化（包含批量归一化）
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.bn8(self.conv8(x))))

        # 将输出转换为(batch_size, num_features, feature_dim)用于自注意力层
        x = x.transpose(1, 2)
        x = self.attention1(x)

        # 重新将输出转换回原来的形状
        x = x.transpose(1, 2)

        # 展平卷积层的输出，使用实例属性
        x = x.reshape(-1, self.fc_input_size)

        # 通过全连接层
        x = F.relu(self.fc1(x))
        # 其他全连接层的操作...
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.attention2(x.view(x.size(0), -1, 1024))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.fc7(x)

        return x