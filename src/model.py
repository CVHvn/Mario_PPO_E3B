import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# copy from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, use_layer_init):
        super(Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=1)
        self.conv2 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = layer_init(nn.Linear(1152, 512)) if use_layer_init else nn.Linear(1152, 512)
        self.in_critic = layer_init(nn.Linear(512, 1)) if use_layer_init else nn.Linear(512, 1)
        self.ex_critic = layer_init(nn.Linear(512, 1)) if use_layer_init else nn.Linear(512, 1)
        self.actor_linear = layer_init(nn.Linear(512, output_dim)) if use_layer_init else nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        return self.actor_linear(x), self.ex_critic(x), self.in_critic(x)

class Embedding_Model(nn.Module):
    def __init__(self, embedding_dim, use_layer_init):
        super(Embedding_Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(1, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear1 = layer_init(nn.Linear(1152, embedding_dim)) if use_layer_init else nn.Linear(1152, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x

class InverseDynamicsModel(nn.Module):
    def __init__(self, embedding_dim, num_action, use_layer_init):
        super(InverseDynamicsModel, self).__init__()
        self.linear1 = layer_init(nn.Linear(embedding_dim * 2, 256)) if use_layer_init else nn.Linear(embedding_dim * 2, 256)
        self.linear2 = layer_init(nn.Linear(256, num_action)) if use_layer_init else nn.Linear(256, num_action)

    def forward(self, emb_t, emb_t_):
        x = torch.cat([emb_t, emb_t_], dim=-1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)