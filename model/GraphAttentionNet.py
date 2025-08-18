import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphAttentionLayer(nn.Module):
    def __init__(self,embedding_dim, n_heads):
        super(GraphAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)
        self.b_n = nn.BatchNorm1d(embedding_dim)
        self.feedforward= nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        tmp = x
        x = self.mha(x,x,x)[0]
        x += tmp
        x = self.b_n(x.permute(0,2,1)).permute(0,2,1)
        tmp = x
        x = self.feedforward(x)
        x = F.leaky_relu(x)
        x+= tmp
        x = self.b_n(x.permute(0,2,1)).permute(0,2,1)
        return x


"""
graph features encoder 
"""
class GraphAttentionEncoder(nn.Module):
    def __init__(self, batch_size, embedding_dim, n_nodes, n_layers):
        super(GraphAttentionEncoder, self).__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.input_embedding = nn.Linear(3, embedding_dim)
        self.attention_layers = nn.ModuleList()
        for i in range(n_layers):
            self.attention_layers.append(GraphAttentionLayer(embedding_dim, 8))
    def forward(self, x):
        x = self.input_embedding(x)
        for i in range(self.n_layers):
            x = self.attention_layers[i](x)
        return x


"""
path planning decoder 
"""
class PointerDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_output, max_steps=None):
        # encoder_output: [batch, num_nodes, embed_dim]
        batch, num_nodes, embed_dim = encoder_output.size()
        device = encoder_output.device

        # 初始化隐藏状态
        h = torch.zeros(batch, self.hidden_dim).to(device)
        c = torch.zeros(batch, self.hidden_dim).to(device)

        # 选取起始节点
        start = torch.zeros(batch, embed_dim).to(device)
        visited_mask = torch.zeros(batch, num_nodes).to(device)
        seq = []
        logp_list = []
        if max_steps is None:
            max_steps = num_nodes

        for _ in range(max_steps):
            h, c = self.lstm(start, (h, c))  # LSTM更新隐藏状态

            # 计算注意力分数
            q = self.W_q(h).unsqueeze(1)           # [batch,1,hidden]
            k = self.W_k(encoder_output)           # [batch,num_nodes,hidden]
            scores = self.v(torch.tanh(q + k)).squeeze(-1)  # [batch, num_nodes]

            # 屏蔽已访问节点
            scores = scores - 1e6 * visited_mask

            probs = F.softmax(scores, dim=-1)      # 概率分布
            idx = torch.multinomial(probs, 1).squeeze(-1)  # 采样

            action_probs = probs[torch.arange(batch), idx]
            logp_list.append(action_probs)

            # 更新 mask
            visited_mask[torch.arange(batch), idx] = 1

            # 保存选择序列
            seq.append(idx)

            # 准备下一步输入
            start = encoder_output[torch.arange(batch), idx]

        # 输出 shape: [batch, num_nodes]
        action_sequence = torch.stack(seq, dim=1)
        # 概率 shape: [batch]
        logp_list = torch.stack(logp_list, dim=1)
        logp_sum = torch.sum(logp_list, dim=1)
        return action_sequence, logp_sum
