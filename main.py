import json

import torch
import argparse
from model import Agent
from utils import generate_data

save_num = 3
def get_args() -> argparse.Namespace:
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
    parser.add_argument('--train', type=bool, default=True, help='Train or evaluate' )
    return parser.parse_args()

def train() -> None:
    args = get_args()
    agent = Agent(batch_size=args.batch_size, embedding_dim=args.embedding_dim,
                       n_nodes=args.n_nodes, n_layers=args.n_layer, hidden_dim=args.hidden_dim,
                       hidden_layer_num=args.hidden_layer_num, lr=args.lr)
    a_loss_list = []
    c_loss_list = []
    r_mean_list = []
    agent.train()
    log_dict =  {}

    for epoch in range(args.epochs):
        graph_instance = generate_data(args.batch_size, args.n_nodes, args.grid_edge, args.train)
        a_loss, c_loss, r_mean = agent.train_model(graph_instance)
        a_loss_list.append(a_loss)
        c_loss_list.append(c_loss)
        r_mean_list.append(r_mean)
        if (epoch + 1) % (args.epochs / 100) == 0:
            print(f"train on {epoch + 1} epochs")
            log_dict[f"record{int((epoch + 1) / (args.epochs / 100))}"] = {"actor_loss": sum(a_loss_list) / len(a_loss_list),
                                                                      "critic_loss": sum(c_loss_list) / len(c_loss_list),
                                                                      "reward_mean": sum(r_mean_list) / len(r_mean_list)
                                                                      }
            a_loss_list.clear()
            c_loss_list.clear()
            r_mean_list.clear()

    log_json = json.dumps(log_dict, indent=4)
    with open(f'./log/logs{save_num}.json', 'w') as f:
        f.write(log_json)

    torch.save(agent.state_dict(), f'./checkpoints/agent{save_num}.pt')
    print("training has been finished")

def evaluate() -> None:
    pass


def main() -> None:
    train()
    # evaluate()




if __name__ == '__main__':
    main()
