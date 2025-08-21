import torch
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


class Critic(nn.Module):
    """
    Critic network
    return state-value
    """
    def __init__(self, batch_size, n_nodes, n_layers, embedding_dim, hidden_dim, hidden_layer_num):
        super(Critic, self).__init__()
        self.mlp_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU())
        self.encoder = GraphAttentionEncoder(batch_size=batch_size, embedding_dim=embedding_dim, n_nodes=n_nodes, n_layers=n_layers)
        self.value_net = nn.Sequential(*(self.mlp_layer for _ in range(hidden_layer_num)))
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x,dim=1).squeeze()
        x = self.value_net(x)
        x = self.output_layer(x).squeeze()
        return x

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
        self.critic = Critic(batch_size, n_nodes, n_layers, embedding_dim, hidden_dim, hidden_layer_num)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, graph_instance):
        graph_instance = torch.tensor(graph_instance, dtype=torch.float32, device='cuda')
        action_sequence, log_sum = self.actor(graph_instance)
        return action_sequence, log_sum

    def evaluate_state(self, graph_instance):
        graph_instance = torch.tensor(graph_instance, dtype=torch.float32, device='cuda')
        value = self.critic(graph_instance)
        return value

    def train_model(self, graph_instance):
        action_seq, log_sum = self.take_action(graph_instance)
        rewards = cal_reward(graph_instance, action_seq.detach().cpu().numpy())
        value = self.evaluate_state(graph_instance)
        advantage = cal_advantage(rewards, value)
        advantage = advantage_normalization(advantage)
        actor_loss = (-log_sum * advantage).mean(dim=0)
        critic_loss = F.mse_loss(value, rewards)
        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
        self.actor_optimizer.step()
        # critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)
        self.critic_optimizer.step()
        return actor_loss.item(), critic_loss.item(), rewards.mean().item()




