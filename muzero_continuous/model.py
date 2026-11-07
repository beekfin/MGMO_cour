"""Три функции модели; обучение и поиск на данном этапе ещё не подключены."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Prediction:
    hidden: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor | None


class MuZeroModel(nn.Module):
    def __init__(self, observation_size=3, action_count=5, hidden_size=32, action_scale=2.0):
        super().__init__()
        if min(observation_size, action_count, hidden_size) < 1 or action_scale <= 0:
            raise ValueError("Размеры и масштаб должны быть положительными")
        self.action_scale = float(action_scale)
        self.representation = nn.Sequential(
            nn.Linear(observation_size, 64), nn.ReLU(),
            nn.Linear(64, hidden_size), nn.Tanh())
        self.dynamics_body = nn.Sequential(nn.Linear(hidden_size+1,64), nn.ReLU())
        self.next_hidden = nn.Sequential(nn.Linear(64,hidden_size), nn.Tanh())
        self.reward = nn.Linear(64,1)
        self.prediction_body = nn.Sequential(nn.Linear(hidden_size,64), nn.ReLU())
        self.policy = nn.Linear(64,action_count)
        self.value = nn.Linear(64,1)

    def predict(self, hidden, reward=None):
        features = self.prediction_body(hidden)
        # Награда и ценность имеют неограниченные скалярные выходы.
        return Prediction(hidden, self.policy(features), self.value(features).squeeze(-1), reward)

    def initial_inference(self, observation):
        return self.predict(self.representation(observation))

    def recurrent_inference(self, hidden, numeric_action):
        if numeric_action.ndim != 2 or numeric_action.shape != (hidden.shape[0],1):
            raise ValueError("Действие должно иметь форму (batch, 1)")
        features = self.dynamics_body(torch.cat((hidden, numeric_action/self.action_scale), dim=-1))
        return self.predict(self.next_hidden(features), self.reward(features).squeeze(-1))
