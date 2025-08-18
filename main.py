import torch
import argparse
from model import Agent
from utils import generate_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_nodes', type=int, default=11, help='Number of nodes in TSP')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--hidden_layer_num', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--grid_edge', type=int, default=4000, help='Grid edge length')
    return parser.parse_args()


def train():
    args = get_args()
    agent = Agent(batch_size=args.batch_size, embedding_dim=args.embedding_dim,
                       n_nodes=args.n_nodes, n_layers=args.n_layer, hidden_dim=args.hidden_dim,
                       hidden_layer_num=args.hidden_layer_num, lr=args.lr)
    agent.train()
    for epoch in range(args.epochs):
        print(f"train on {epoch} epochs")
        graph_instance = generate_data(args.batch_size, args.n_nodes, args.grid_edge)
        agent.train_model(filename='./log/logs.txt', instance=graph_instance)


def eval_model():
    pass


def main():
    train()
    # eval_model()




if __name__ == '__main__':
    main()
