import json
import torch
import argparse
from model import Agent
from baseline import cities_form_transfer, TSPSolver
from utils import generate_data, plot_original, plot_results, plot_log

save_num = 2

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_nodes', type=int, default=11, help='Number of nodes in TSP')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--hidden_layer_num', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--step', type=int, default=12, help='Number of steps')
    parser.add_argument('--grid_edge', type=int, default=4, help='Grid edge length')
    parser.add_argument('--train', type=bool, default=True, help='Train or evaluate' )
    return parser.parse_args()

def train() -> None:
    args = get_args()
    agent = Agent(batch_size=args.batch_size, embedding_dim=args.embedding_dim,
                       n_nodes=args.n_nodes, n_layers=args.n_layer, hidden_dim=args.hidden_dim,
                       hidden_layer_num=args.hidden_layer_num, lr=args.lr)
    # agent.load_state_dict(torch.load(f'./checkpoints/agent{save_num-1}.pt', map_location=torch.device('cuda')))
    agent.to('cuda')
    a_loss_list = []
    c_loss_list = []
    r_mean_list = []
    log_dict =  {}
    agent.train()


    for epoch in range(args.epochs):
        graph_instance = generate_data(args.batch_size, args.n_nodes, args.grid_edge, args.train)
        for step in range(args.step):
            a_loss, c_loss, r_mean = agent.train_model(graph_instance)
            a_loss_list.append(a_loss)
            c_loss_list.append(c_loss)
            r_mean_list.append(r_mean)
            if (epoch * args.step + (step + 1)) % (args.step * args.epochs / 100) == 0:
                print(f'train on {(epoch * args.step + (step + 1)) / (args.step * args.epochs / 100)}%')
                log_dict[f"record{int((epoch * args.step + (step + 1)) / (args.step * args.epochs / 100))}"] = {
                    "actor_loss": sum(a_loss_list) / len(a_loss_list),
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
    args = get_args()
    state_dict = torch.load(f'./checkpoints/agent{save_num}.pt', map_location=torch.device('cuda'))
    agent = Agent(batch_size=args.batch_size, embedding_dim=args.embedding_dim,
                       n_nodes=args.n_nodes, n_layers=args.n_layer, hidden_dim=args.hidden_dim,
                       hidden_layer_num=args.hidden_layer_num, lr=args.lr)
    agent.load_state_dict(state_dict)
    agent.to('cuda')
    graph_instance = generate_data(args.batch_size, args.n_nodes, args.grid_edge, False)
    seq_action, _ = agent.take_action(graph_instance)
    plot_original(graph_instance)
    plot_results(graph_instance, seq_action.detach().cpu().numpy())
    plot_log(f'./log/logs{save_num}.json')

def valid() -> None:
    solver = TSPSolver(pop_size=100, max_iter=800, mutation_rate=0.1, verbose=True)
    cities = cities_form_transfer('./datalib/graph_instance_json.json')
    solution_list = []
    for batch in range(len(cities)):
        solution = solver.solve(cities[f"batch{batch}"])
        solution_list.append(solution)




def main() -> None:
    train()
    evaluate()
    # valid()

if __name__ == '__main__':
    main()
