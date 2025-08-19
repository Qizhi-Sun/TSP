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
        self.encoder = GraphAttentionEncoder(batch_size=batch_size,embedding_dim=embedding_dim,n_nodes=n_nodes, n_layers=n_layers)
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
        self.embedding_layer = nn.Linear(3, embedding_dim)
        self.input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.value_net = nn.Sequential(*(self.mlp_layer for _ in range(hidden_layer_num)))
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.value_net(x)
        x = self.output_layer(x)
        return x

def reward_normalization(reward_batch:torch.Tensor) -> torch.Tensor:
    """
    :param reward_batch: rewards
    :return: normalized rewards
    """
    reward_mean = reward_batch.mean()
    reward_std = reward_batch.std()
    reward_batch = (reward_batch - reward_mean) / reward_std
    return reward_batch

def cal_reward(graph_instance:np.ndarray, action_seq:np.ndarray) -> torch.Tensor:
    """
    :param action_seq: sequence actions
    :param graph_instance: instance of graph
    :return: rewards
    """
    rewards = compute_length(graph_instance, action_seq)
    rewards = torch.tensor(rewards, dtype=torch.float32)
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

def rewards_log(func):
    def wrapper(*args, **kwargs):
        actor_loss, critic_loss, r_mean = func(args[0], kwargs['instance'])
        with open(kwargs['filename'], 'a') as f:
            f.write('actor_loss: {}, critic_loss: {}, r_mean: {}\n'.format(actor_loss, critic_loss, r_mean))
    return wrapper


class Agent(nn.Module):
    def __init__(self, batch_size, embedding_dim, n_nodes, n_layers, hidden_dim, hidden_layer_num, lr):
        super(Agent, self).__init__()
        self.actor = Actor(batch_size, embedding_dim, n_nodes, n_layers, hidden_dim)
        self.critic = Critic(batch_size, n_nodes, n_layers, embedding_dim, hidden_dim, hidden_layer_num)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, graph_instance):
        graph_instance = torch.tensor(graph_instance, dtype=torch.float32)
        action_sequence, log_sum = self.actor(graph_instance)
        return action_sequence, log_sum

    def evaluate_state(self, graph_instance):
        graph_instance = torch.tensor(graph_instance, dtype=torch.float32)
        graph_instance = graph_instance.mean(dim=1)
        value = self.critic(graph_instance).squeeze(-1)
        return value

    def train_model(self, graph_instance):
        action_seq, log_sum = self.take_action(graph_instance)
        rewards = cal_reward(graph_instance, action_seq.detach().cpu().numpy())
        rewards = reward_normalization(rewards)
        value = self.evaluate_state(graph_instance)
        advantage = cal_advantage(rewards, value)
        actor_loss = (-log_sum * advantage).mean(dim=0)
        critic_loss = F.mse_loss(value, rewards)
        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        x = rewards.mean()
        return actor_loss.item(), critic_loss.item(), rewards.mean().item()




