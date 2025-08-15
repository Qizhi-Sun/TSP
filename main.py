import torch
from model import GraphAttentionEncoder, PointerDecoder


def main():
    x = torch.randn(64, 11, 3)
    model = GraphAttentionEncoder(batch_size=64, embedding_dim=256, n_nodes=11, n_layers=8)
    tmp = model(x)
    model1 = PointerDecoder(embed_dim=256, hidden_dim=256)
    results, lp = model1(tmp)
    print(results)





if __name__ == '__main__':
    main()
