import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RoleModel(nn.Module):
    def __init__(self, init_features, dr_rate, class_num):
        super().__init__()
        NODE_NUM = init_features.shape[0]
        emb_size = init_features.shape[1]
        
        HIDDEN_SIZES = [emb_size, emb_size//2, emb_size//4]
        
        self.embed = nn.Embedding(NODE_NUM, HIDDEN_SIZES[0])
        self.embed.weight.data.copy_(torch.from_numpy(init_features))
        self.linear1 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.linear3 = nn.Linear(HIDDEN_SIZES[2], class_num)

    def forward(self, X):
        X= self.embed(X)
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.log_softmax(self.linear3(X))
        return X


class Discriminator(nn.Module):
    def __init__(self, dr_rate, emb_size):
        super().__init__()
        HIDDEN_SIZES = [emb_size, emb_size//2, emb_size//4]
        
        self.linear1 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.linear3 = nn.Linear(HIDDEN_SIZES[2], 1)
  
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        X = F.sigmoid(self.linear3(X))
        return X


class SingleRoleModel(nn.Module):
    def __init__(self, dr_rate, emb_size, class_num):
        super().__init__()
        HIDDEN_SIZES = [emb_size, emb_size//2, emb_size//4]
        
        self.linear1 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.dr_rate = dr_rate
        self.linear2 = nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2])
        self.linear3 = nn.Linear(HIDDEN_SIZES[2], class_num)
  
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.relu(self.linear2(X))
        X = F.dropout(X, self.dr_rate, training=self.training)
        X = F.log_softmax(self.linear3(X))
        return X
