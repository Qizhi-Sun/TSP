import torch
import copy
from .GraphAttentionNet import *
import numpy as np
from utils import compute_length


class Actor(nn.Module):
    """
    Actor network
    return batch_action_seq & log_prob_sum
    """
    def __init__(self, batch_size, embedding_dim, n_nodes, n_layers, hidden_dim):
        super(Actor, self).__init__()
        self.encoder = GraphAttentionEncoder(batch_size=batch_size,embedding_dim=embedding_dim, n_nodes=n_nodes, n_layers=n_layers)
        self.decoder = PointerDecoder(embed_dim=embedding_dim, hidden_dim=hidden_dim)
    def forward(self, graph_instance):
        graph_features = self.encoder(graph_instance)
        action_sequence, log_sum = self.decoder(graph_features)
        return action_sequence, log_sum

class Rollout:
    def __init__(self, actor, alpha = 0.95):
        self.baseline_actor = copy.deepcopy(actor).to('cuda')
        self.alpha = alpha
        for p in self.baseline_actor.parameters():
            p.requires_grad = False

    def update(self, actor):
        for param, baseline_param in zip(actor.parameters(), self.baseline_actor.parameters()):
            baseline_param.data.copy_(self.alpha * baseline_param.data + (1 - self.alpha) * param.data)

    def evaluate(self, graph_instance):
        graph_instance = torch.tensor(graph_instance, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            action_seq, _ = self.baseline_actor(graph_instance)
        return action_seq

def advantage_normalization(advantage:torch.Tensor) -> torch.Tensor:
    """
    :param advantage: advantage
    :return: normalized rewards
    """
    advantage_mean = advantage.mean()
    advantage_std = advantage.std()
    advantage_norm = (advantage - advantage_mean) / (advantage_std + 1e-5)
    return advantage_norm

def cal_reward(graph_instance:np.ndarray, action_seq:np.ndarray) -> torch.Tensor:
    """
    :param action_seq: sequence actions
    :param graph_instance: instance of graph
    :return: rewards
    """
    rewards = compute_length(graph_instance, action_seq)
    rewards = torch.tensor(rewards, dtype=torch.float32, device='cuda')
    return rewards


def cal_advantage(reward_batch:torch.Tensor, value_batch:torch.Tensor) -> torch.Tensor:
    """
    :param reward_batch: rewards
    :param value_batch: value calculated by critic
    :return: advantage
    """
    delta_batch = reward_batch - value_batch
    advantage_batch = delta_batch.detach()
    return advantage_batch


class Agent(nn.Module):
    def __init__(self, batch_size, embedding_dim, n_nodes, n_layers, hidden_dim, hidden_layer_num, lr):
        super(Agent, self).__init__()
        self.actor = Actor(batch_size, embedding_dim, n_nodes, n_layers, hidden_dim)
        self.rollout = Rollout(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def take_action(self, graph_instance):
        graph_instance = torch.tensor(graph_instance, dtype=torch.float32, device='cuda')
        action_sequence, log_sum = self.actor(graph_instance)
        return action_sequence, log_sum

    def train_model(self, graph_instance):
        action_seq, log_sum = self.take_action(graph_instance)
        baseline_action_seq = self.rollout.evaluate(graph_instance)
        rewards = cal_reward(graph_instance, action_seq.detach().cpu().numpy())
        value = cal_reward(graph_instance, baseline_action_seq.detach().cpu().numpy())
        advantage = cal_advantage(rewards, value)
        advantage = advantage_normalization(advantage)
        actor_loss = (log_sum * advantage).mean(dim=0)
        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
        self.actor_optimizer.step()
        # rollout update
        self.rollout.update(self.actor)
        return actor_loss.item(), 0, rewards.mean().item()




